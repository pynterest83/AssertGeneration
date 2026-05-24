"""End-to-end pipeline runner that wraps src/solution code with SSE progress.

Flow:
  1. extract_tests → writes infer_input/{inputs,meta_llm}.csv
  2. CodeGraph(...) builds/loads KùzuDB graph (with on_progress callback)
  3. For each test case: build_graph(llm, store) + stream → emit per-agent + per-sample events,
     write incremental rows to infer_input/oracle_preds.csv (resume-safe)
  4. inject_tests → applies predictions back into test files
"""
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from progress import push_progress
from test_extractor import extract_tests
from injectors import inject_tests

# Imports from src/solution/ (sys.path injected by server.py)
from code_graph import CodeGraph  # noqa: E402
from graph import build_graph  # noqa: E402
from helpers.assertion_utils import post_process_assertion  # noqa: E402
from helpers.parsing_utils import extract_focal_class, extract_return_type  # noqa: E402
from run_pipeline import CSV_FIELDS, load_done, result_to_row  # noqa: E402
from tools.definitions import is_quota_error  # noqa: E402

log = logging.getLogger("assertgen")


# ---------------------------------------------------------------------------
# CodeGraph cache (per project_path)
# ---------------------------------------------------------------------------

_graph_cache: dict[str, CodeGraph] = {}
_graph_lock = threading.Lock()


def get_or_build_graph(project_path: str, language: str,
                       force_reindex: bool = False,
                       on_progress=None) -> CodeGraph:
    """Get cached CodeGraph or build a new one. Thread-safe."""
    with _graph_lock:
        if not force_reindex and project_path in _graph_cache:
            return _graph_cache[project_path]
        store = CodeGraph(
            project_path,
            language=language,
            force_reindex=force_reindex,
            on_progress=on_progress,
        )
        _graph_cache[project_path] = store
        return store


def has_graph_built(project_path: str) -> bool:
    if project_path in _graph_cache:
        return True
    return (Path(project_path) / ".code_graph.complete").exists() or \
           (Path(project_path) / ".code_graph").exists()


# ---------------------------------------------------------------------------
# Item construction from extracted CSVs
# ---------------------------------------------------------------------------

def build_items(project_path: str, language: str) -> list[dict]:
    """Load inputs.csv + meta_llm.csv produced by test_extractor and build the
    item dicts that run_pipeline.process_single expects."""
    infer_input = Path(project_path) / "infer_input"
    inputs_df = pd.read_csv(str(infer_input / "inputs.csv"))
    meta_df = pd.read_csv(str(infer_input / "meta_llm.csv"))

    if "test_name" in inputs_df.columns and "test_name" in meta_df.columns:
        merged_df = inputs_df.merge(meta_df, on="test_name", how="inner",
                                    suffixes=("", "_meta"))
    else:
        if len(inputs_df) != len(meta_df):
            raise ValueError(
                f"inputs.csv ({len(inputs_df)} rows) and meta_llm.csv "
                f"({len(meta_df)} rows) have different row counts."
            )
        extra_cols = [c for c in meta_df.columns if c not in inputs_df.columns]
        merged_df = pd.concat(
            [inputs_df.reset_index(drop=True),
             meta_df[extra_cols].reset_index(drop=True)],
            axis=1,
        )

    items: list[dict] = []
    for _, row in merged_df.iterrows():
        focal_method = str(row["focal_method"])
        docstring = str(row.get("docstring", "")) if pd.notna(row.get("docstring")) else ""
        test_prefix = str(row["test_prefix"]) if pd.notna(row.get("test_prefix")) else ""
        gt_output = str(row.get("GT_output", "")) if pd.notna(row.get("GT_output")) else ""
        file_path = str(row.get("file_path", "")) if pd.notna(row.get("file_path")) else ""
        test_name = str(row.get("test_name", "")) if pd.notna(row.get("test_name")) else ""
        return_type = extract_return_type(focal_method) or ""

        items.append({
            "focal_method": focal_method,
            "focal_class": extract_focal_class(test_name, language),
            "language": language,
            "docstring": docstring,
            "test_prefix": test_prefix,
            "return_type": return_type,
            "test_name": test_name,
            "file_path": file_path,
            "gt_output": gt_output,
        })
    return items


# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------

def make_llm(api_endpoint: str, model_name: str, api_key: str,
             temperature: float, max_tokens: int = 4096) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=api_endpoint,
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=1,
        timeout=120,
        http_client=httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)
        ),
        http_async_client=httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)
        ),
        streaming=False,
    )


def check_api(llm: ChatOpenAI) -> str | None:
    """Return None on success, error message on failure. Does NOT exit."""
    try:
        llm.invoke([HumanMessage(content="hi")])
    except Exception as e:
        return f"API connection failed: {e}"
    return None


# ---------------------------------------------------------------------------
# Per-sample inference with per-agent SSE events
# ---------------------------------------------------------------------------

def _initial_state(item: dict) -> dict:
    return {
        "focal_method": item["focal_method"],
        "focal_class": item.get("focal_class", ""),
        "language": item.get("language", "java"),
        "docstring": item.get("docstring", ""),
        "test_prefix": item["test_prefix"],
        "return_type": item.get("return_type", ""),
        "test_name": item.get("test_name", ""),
        "file_path": item.get("file_path", ""),
        "is_exception": False,
        "exception_reasoning": "",
        "analysis": "",
        "prediction": "",
        "assertion": "",
    }


def _process_one(llm, store, language: str, item: dict) -> dict:
    """Stream the compiled graph for one item, emitting per-agent events.

    Returns the result dict (item enriched with assertion/is_exception)."""
    compiled = build_graph(llm, store, language=language)
    final_state: dict | None = None
    try:
        for step in compiled.stream(
            _initial_state(item),
            config={"run_name": item.get("test_name", ""), "tags": ["extension"]},
        ):
            for node_name, node_state in step.items():
                push_progress({
                    "type": "inference",
                    "agent": node_name,
                    "status": "started",
                    "test_name": item.get("test_name", ""),
                })
                final_state = node_state
    except Exception as e:
        if is_quota_error(e):
            raise
        log.warning("Stream failed for %s: %s", item.get("test_name", "?"), e)
        return {**item, "assertion": "", "is_exception": False}

    state = final_state or {}
    is_exception = bool(state.get("is_exception", False))
    raw_assertion = state.get("assertion", "")
    assertion = "exception" if is_exception else post_process_assertion(
        raw_assertion, item.get("language", "java")
    )
    return {**item, "assertion": assertion, "is_exception": is_exception}


# ---------------------------------------------------------------------------
# Incremental CSV writer (mirrors run_pipeline.run_inference logic)
# ---------------------------------------------------------------------------

def _open_writer(output_csv: Path, append: bool):
    import csv as _csv
    fh = open(output_csv, "a" if append else "w", newline="", encoding="utf-8")
    writer = _csv.DictWriter(fh, fieldnames=CSV_FIELDS, quoting=_csv.QUOTE_ALL)
    if not append:
        writer.writeheader()
    return fh, writer


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def execute_pipeline(req) -> None:
    """Run the full pipeline. `req` is a Pydantic PipelineConfig (duck-typed below).

    Streams progress via push_progress(). Errors are caught and emitted as
    `pipeline_error` events; callers can subscribe to /progress to receive them.
    """
    project_path = req.project_path
    language = req.language
    t_total = time.time()

    try:
        # ── Stage 1: Extract ────────────────────────────────────────────────
        push_progress({"type": "stage", "stage": "extraction",
                       "message": "Extracting test cases..."})
        t0 = time.time()
        extract_result = extract_tests(project_path, language, push_progress)
        test_count = extract_result["test_count"]
        push_progress({"type": "extraction_complete", "test_count": test_count,
                       "inputs_csv": extract_result["inputs_csv"],
                       "meta_csv": extract_result["meta_csv"]})
        log.info("Extract done in %.1fs — %d cases", time.time() - t0, test_count)

        if test_count == 0:
            push_progress({"type": "pipeline_complete", "total_tests": 0,
                           "injected_files": [],
                           "message": "No test cases found."})
            return

        # ── Stage 2: Build graph ───────────────────────────────────────────
        push_progress({"type": "stage", "stage": "graph_building",
                       "message": "Building code graph..."})
        t0 = time.time()

        def graph_cb(cur: int, tot: int, fp: str):
            push_progress({"type": "graph_building", "phase": "parsing",
                           "current": cur, "total": tot, "file": fp})

        store = get_or_build_graph(
            project_path, language,
            force_reindex=req.force_reindex,
            on_progress=graph_cb,
        )
        push_progress({"type": "graph_building_complete"})
        log.info("Graph done in %.1fs", time.time() - t0)

        # ── Stage 3: Inference ─────────────────────────────────────────────
        push_progress({"type": "stage", "stage": "inference",
                       "message": "Running LLM inference..."})
        t0 = time.time()

        llm = make_llm(req.api_endpoint, req.model_name, req.api_key,
                       req.temperature)
        err = check_api(llm)
        if err:
            push_progress({"type": "pipeline_error", "message": err})
            return

        items = build_items(project_path, language)
        total = len(items)

        output_csv = Path(project_path) / "infer_input" / "oracle_preds.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        done_names = load_done(str(output_csv))
        items_todo = [it for it in items if it.get("test_name", "") not in done_names]

        append = output_csv.exists() and bool(done_names)
        fh = writer = None
        write_lock = threading.Lock()
        if items_todo:
            fh, writer = _open_writer(output_csv, append)

        completed = len(done_names)
        stop_event = threading.Event()
        results: list[dict] = []

        def worker(item: dict) -> dict:
            if stop_event.is_set():
                return {**item, "assertion": "", "is_exception": False, "skipped": True}
            return _process_one(llm, store, language, item)

        try:
            with ThreadPoolExecutor(max_workers=req.max_workers) as ex:
                futures = {ex.submit(worker, it): it for it in items_todo}
                for fut in as_completed(futures):
                    item = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        if is_quota_error(e):
                            if not stop_event.is_set():
                                stop_event.set()
                                push_progress({
                                    "type": "pipeline_error",
                                    "message": f"Quota exhausted: {e}",
                                })
                            res = {**item, "assertion": "", "is_exception": False,
                                   "skipped": True}
                        else:
                            log.warning("Future failed for %s: %s",
                                        item.get("test_name", "?"), e)
                            res = {**item, "assertion": "", "is_exception": False}
                    results.append(res)
                    completed += 1
                    push_progress({
                        "type": "inference_progress",
                        "current": completed,
                        "total": total,
                        "test_name": item.get("test_name", ""),
                    })
                    if writer and not res.get("skipped"):
                        with write_lock:
                            writer.writerow(result_to_row(res))
                            fh.flush()
        finally:
            if fh:
                fh.close()

        push_progress({"type": "inference_complete", "result_count": len(results)})
        log.info("Inference done in %.1fs — %d results", time.time() - t0, len(results))

        # ── Stage 4: Inject ────────────────────────────────────────────────
        push_progress({"type": "stage", "stage": "injection",
                       "message": "Injecting assertions..."})
        t0 = time.time()
        injected_files: list[str] = []
        try:
            injected_files = inject_tests(language, project_path, str(output_csv)) or []
            for f in injected_files:
                push_progress({"type": "injection", "file": f, "count": 1})
        except Exception as e:
            log.warning("Injection warning: %s", e)
            push_progress({"type": "injection_warning", "message": str(e)})
        log.info("Inject done in %.1fs — %d files", time.time() - t0, len(injected_files))

        push_progress({
            "type": "pipeline_complete",
            "total_tests": total,
            "injected_files": injected_files,
        })
        log.info("Total pipeline: %.1fs", time.time() - t_total)

    except Exception as e:
        log.error("Pipeline crashed: %s", e)
        traceback.print_exc()
        push_progress({"type": "pipeline_error", "message": str(e)})

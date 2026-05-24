"""FastAPI server for AssertGen VSCode extension.

Wraps src/solution/ as the inference engine. Resolves the solution path in
this order so it works both in dev (uvicorn from src/extension/backend) and
when bundled into the extension (frontend/backend/_solution):
    1. <here>/_solution     (bundled mode)
    2. <here>/../solution   (dev mode: src/extension/backend → src/solution)
    3. <here>/../../solution
"""
import json
import logging
import queue
import sys
import threading
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve src/solution path BEFORE any imports that depend on it
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "_solution",
    _HERE.parent / "solution",
    _HERE.parent.parent / "solution",
]
for _p in _CANDIDATES:
    if _p.exists() and (_p / "run_pipeline.py").exists():
        sys.path.insert(0, str(_p))
        _SOLUTION_DIR = _p
        break
else:
    raise RuntimeError(
        f"Could not locate src/solution directory. Tried: {_CANDIDATES}"
    )

import pandas as pd  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from progress import get_queue, push_progress  # noqa: E402
import pipeline_runner  # noqa: E402
from graph_export import export_graph_json  # noqa: E402
from test_extractor import extract_tests  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("assertgen")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

log.info("Solution dir: %s", _SOLUTION_DIR)

app = FastAPI(title="AssertGen Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    project_path: str
    language: str = "python"


class BuildGraphRequest(BaseModel):
    project_path: str
    language: str = "python"
    force_reindex: bool = False


class PipelineConfig(BaseModel):
    project_path: str
    language: str = "python"
    api_endpoint: str
    model_name: str
    api_key: str = "EMPTY"
    max_workers: int = 8
    temperature: float = 0.0
    force_reindex: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract_endpoint(req: ExtractRequest):
    log.info("POST /extract  project=%s  lang=%s", req.project_path, req.language)
    try:
        result = extract_tests(req.project_path, req.language, push_progress)
        log.info("Extraction: %d test cases", result["test_count"])
        return {"status": "ok", **result}
    except Exception as exc:
        log.error("Extraction failed: %s", exc)
        traceback.print_exc()
        return {"status": "error", "message": str(exc)}


@app.post("/build-graph")
def build_graph_endpoint(req: BuildGraphRequest):
    from helpers.db_utils import Queries
    log.info("POST /build-graph  project=%s  lang=%s  force=%s",
             req.project_path, req.language, req.force_reindex)

    def cb(cur: int, tot: int, fp: str):
        push_progress({"type": "graph_building", "phase": "parsing",
                       "current": cur, "total": tot, "file": fp})

    try:
        import time
        t0 = time.time()
        store = pipeline_runner.get_or_build_graph(
            req.project_path, req.language,
            force_reindex=req.force_reindex,
            on_progress=cb,
        )
        elapsed = time.time() - t0
        try:
            method_count = store.conn.execute(
                Queries.COUNT_METHODS_EXCLUDING_SENTINEL
            ).get_next()[0]
            class_count = store.conn.execute(Queries.COUNT_CLASSES).get_next()[0]
        except Exception:
            method_count = class_count = 0
        log.info("Graph built in %.1fs — %d classes, %d methods",
                 elapsed, class_count, method_count)
        return {"status": "ok",
                "class_count": class_count,
                "method_count": method_count}
    except Exception as exc:
        log.error("Graph build failed: %s", exc)
        traceback.print_exc()
        return {"status": "error", "message": str(exc)}


@app.get("/graph-data")
def graph_data_endpoint(project_path: str, language: str = "python"):
    log.info("GET /graph-data  project=%s", project_path)
    try:
        store = pipeline_runner.get_or_build_graph(
            project_path, language, force_reindex=False
        )
        data = export_graph_json(store)
        log.info("Graph data: %d nodes, %d edges",
                 len(data.get("nodes", [])), len(data.get("edges", [])))
        return {"status": "ok", **data}
    except Exception as exc:
        log.error("Graph data failed: %s", exc)
        traceback.print_exc()
        return {"status": "error", "message": str(exc),
                "nodes": [], "edges": []}


@app.get("/graph-status")
def graph_status_endpoint(project_path: str):
    return {"built": pipeline_runner.has_graph_built(project_path)}


@app.get("/pipeline-status")
def pipeline_status_endpoint(project_path: str):
    preds_csv = Path(project_path) / "infer_input" / "oracle_preds.csv"
    if not preds_csv.exists():
        return {"has_results": False, "test_count": 0}
    try:
        df = pd.read_csv(preds_csv)
        test_count = len(df)
    except Exception:
        test_count = 0
    return {"has_results": True, "test_count": test_count}


@app.post("/run-pipeline")
def run_pipeline_endpoint(req: PipelineConfig):
    log.info("POST /run-pipeline  project=%s  lang=%s  model=%s  workers=%d",
             req.project_path, req.language, req.model_name, req.max_workers)
    t = threading.Thread(target=pipeline_runner.execute_pipeline,
                         args=(req,), daemon=True)
    t.start()
    return {"status": "started",
            "message": "Pipeline running. Subscribe to /progress for updates."}


@app.get("/progress")
def progress_endpoint():
    q = get_queue()

    def event_generator():
        while True:
            try:
                event = q.get(timeout=0.5)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("pipeline_complete", "pipeline_error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")

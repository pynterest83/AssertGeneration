import os
import re
import sys
import csv
import logging
import argparse
import threading
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

logger = logging.getLogger(__name__)

from code_graph import CodeGraph
from graph import build_graph
from tools.definitions import is_quota_error


def post_process_assertion(raw_assertion, language='java'):
    """Post-process a raw assertion from structured output.

    With structured output, the assertion is already extracted — just apply
    safety-net fixes for paren balancing, negative literals, and trailing comments.
    """
    if not raw_assertion:
        return ''

    raw_assertion = raw_assertion.strip()
    if not raw_assertion:
        return ''

    # Remove any markdown/backtick noise the model may still produce
    raw_assertion = re.sub(r'^```\w*\n?', '', raw_assertion)
    raw_assertion = raw_assertion.replace('`', '').strip()
    raw_assertion = raw_assertion.replace('\u201c', '"').replace('\u201d', '"')
    raw_assertion = raw_assertion.replace('\u2018', "'").replace('\u2019', "'")

    # If structured output returned multiple lines, find the assertion line
    # BUG-10: also handle JS/TS expect(...) pattern
    for line in raw_assertion.split('\n'):
        line = line.strip()
        if line.startswith('assert') or line.startswith('expect('):
            raw_assertion = line
            break

    raw_assertion = strip_trailing_comment(raw_assertion)
    return fix_assertion(raw_assertion, language)


def strip_trailing_comment(line):
    """Remove trailing // or /* comment, but only when outside a string literal."""
    in_str = escape = False
    for i, ch in enumerate(line):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str and ch == '/' and i + 1 < len(line) and line[i + 1] in ('/', '*'):
            return line[:i].rstrip()
    return line


NEG_LITERAL = re.compile(r'^-\d+(?:\.\d+)?[LlFfDd]?$')


def wrap_negative_literals(assertion):
    """Wrap bare negative number literals in parentheses: -2L -> (-2L)."""
    match = re.match(r'(assert\w+\()(.+)(\);)$', assertion.rstrip())
    if not match:
        return assertion

    prefix, inner, suffix = match.group(1), match.group(2), match.group(3)
    args = split_args(inner)
    changed = False
    for i, arg in enumerate(args):
        stripped = arg.strip()
        if NEG_LITERAL.match(stripped):
            args[i] = f'({stripped})'
            changed = True
        else:
            args[i] = stripped
    if not changed:
        return assertion
    return prefix + ', '.join(args) + suffix


def split_args(s):
    """Split assertion arguments respecting parentheses depth and string literals."""
    args, depth, start = [], 0, 0
    in_str = None
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch in '({':
            depth += 1
        elif ch in ')}':
            depth -= 1
        elif ch == ',' and depth == 0:
            args.append(s[start:i])
            start = i + 1
    args.append(s[start:])
    return args


def count_parens(s):
    """Count '(' and ')' outside of string literals (handles both ' and " quotes)."""
    open_p = close_p = 0
    in_str = None  # None, '"', or "'"
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch == '(':
            open_p += 1
        elif ch == ')':
            close_p += 1
    return open_p, close_p


def count_braces(s):
    """Count '{' and '}' outside of string literals (handles both ' and " quotes)."""
    open_b = close_b = 0
    in_str = None
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch == '{':
            open_b += 1
        elif ch == '}':
            close_b += 1
    return open_b, close_b


def fix_assertion(assertion, language='java'):
    if not assertion:
        return assertion

    assertion = assertion.strip()

    # E2: count parens and braces outside string literals
    open_p, close_p = count_parens(assertion)
    open_b, close_b = count_braces(assertion)

    if '() -> {' in assertion and close_b < open_b:
        missing_b = open_b - close_b
        assertion = assertion.rstrip(';') + '}' * missing_b
        open_p, close_p = count_parens(assertion)
        if close_p < open_p:
            assertion += ')' * (open_p - close_p)
    elif close_p < open_p:
        assertion = assertion.rstrip(';') + ')' * (open_p - close_p)

    if language.lower() != 'python':
        if not assertion.endswith(';'):
            assertion += ';'

    assertion = wrap_negative_literals(assertion)

    return assertion


def extract_focal_class(test_name, language='java'):
    """Extract the class under test from test_name.
    e.g. 'org...ContentWriteProgress_ESTest::test1' -> 'ContentWriteProgress'
    """
    if language != 'java':
        return ''
    if '::' not in test_name:
        return ''
    cls = test_name.split('::')[0].rsplit('.', 1)[-1]
    for suffix in ('_ESTest_scaffolding', '_ESTest'):
        if cls.endswith(suffix):
            cls = cls[:-len(suffix)]
            break
    # Companion fix: preserve Java nested class naming using '$'
    # e.g. Outer_Inner_ESTest -> Outer$Inner
    if '_' in cls and '$' not in cls:
        parts = cls.split('_')
        if len(parts) >= 2 and all(parts):
            cls = parts[0] + '$' + '_'.join(parts[1:])
    return cls


def extract_return_type(code):
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    code = str(code).strip()
    if not code or code.lower() == 'nan':
        return None
    match = re.search(
        r'(?:public|private|protected|static|\s)+\s*(?:<[^>]+>\s*)?'
        r'((?:\w+\.)*\w+(?:<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>)?(?:\[\])*)\s+\w+\s*\(',
        code,
    )
    return match.group(1) if match else None


CSV_FIELDS = ['test_name', 'test_prefix', 'file_path', 'assert_pred']


def result_to_row(r: dict) -> dict:
    return {
        'test_name': r.get('test_name', ''),
        'test_prefix': r.get('test_prefix', ''),
        'file_path': r.get('file_path', ''),
        'assert_pred': 'exception' if r.get('is_exception', False) else r.get('assertion', ''),
    }


def load_done(output_file: str) -> set:
    """Return set of test_names already written to output_file."""
    p = Path(output_file)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(output_file)
        if 'test_name' in df.columns:
            return set(df['test_name'].dropna().astype(str))
    except Exception:
        pass
    return set()


def process_single(llm, code_graph, language, item):
    # Build a fresh graph per sample so each gets its own tool closure (counter + ext_cache).
    # Overhead is ~10-50ms (pure Python StateGraph construction, no LLM calls) — negligible
    # compared to per-sample inference time, and required for correctness with MAX_WORKERS > 1.
    compiled_graph = build_graph(llm, code_graph, language=language)
    initial_state = {
        'focal_method': item['focal_method'],
        'focal_class': item.get('focal_class', ''),
        'language': item.get('language', 'java'),
        'docstring': item.get('docstring', ''),
        'test_prefix': item['test_prefix'],
        'return_type': item.get('return_type', ''),
        'test_name': item.get('test_name', ''),
        'file_path': item.get('file_path', ''),
        'is_exception': False,
        'exception_reasoning': '',
        'analysis': '',
        'prediction': '',
        'assertion': '',
    }

    try:
        result = compiled_graph.invoke(
            initial_state,
            config={
                "run_name": item.get('test_name', ''),
                "tags": ["solution5"],
            },
        )
        is_exception = result.get('is_exception', False)
        assertion = 'exception' if is_exception else post_process_assertion(result.get('assertion', ''), item.get('language', 'java'))
        return {
            **item,
            'assertion': assertion,
            'is_exception': is_exception,
        }
    except Exception as e:
        if is_quota_error(e):
            raise  # propagate to run_inference so it can stop the batch
        logger.warning("Graph invoke failed for %s: %s", item.get('test_name', '?'), e)
        return {**item, 'assertion': '', 'is_exception': False}


def run_inference(llm, code_graph, language, items, max_workers, output_file=None, offset=0, limit=None):
    """Run inference with incremental CSV writing, resume, and quota-stop.

    If output_file is given:
    - Already-completed rows in the file are skipped (resume on restart).
    - Each result is flushed to disk immediately (checkpoint per sample).
    - On quota/auth error, remaining queued work is skipped and progress is preserved.
    """
    # Resume: skip already-done test_names
    done_names: set = set()
    if output_file:
        done_names = load_done(output_file)
        if done_names:
            print(f"[resume] Skipping {len(done_names)} already-completed samples.")
    # Slice the assigned chunk first (so offset/limit always refer to the same items),
    # then filter out already-done ones for resumability.
    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    items_todo = [it for it in items if it.get('test_name', '') not in done_names]

    # Open incremental CSV writer
    fh = writer = write_lock = None
    if output_file and items_todo:
        append = Path(output_file).exists() and bool(done_names)
        mode = 'a' if append else 'w'
        fh = open(output_file, mode, newline='', encoding='utf-8')
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        if not append:
            writer.writeheader()
        write_lock = threading.Lock()

    results = []
    stop_event = threading.Event()

    def process(item):
        if stop_event.is_set():
            return {**item, 'assertion': '', 'is_exception': False, 'skipped': True}
        return process_single(llm, code_graph, language, item)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process, item): item for item in items_todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    if is_quota_error(e):
                        if not stop_event.is_set():
                            stop_event.set()
                            print(f"\n[QUOTA] Quota exhausted: {e}", file=sys.stderr)
                            if output_file:
                                print(f"[QUOTA] Progress saved to {output_file}. "
                                      "Re-run to continue.", file=sys.stderr)
                        result = {**item, 'assertion': '', 'is_exception': False, 'skipped': True}
                    else:
                        logger.warning("Future failed for %s: %s", item.get('test_name', '?'), e)
                        result = {**item, 'assertion': '', 'is_exception': False}

                results.append(result)

                if writer and not result.get('skipped'):
                    with write_lock:
                        writer.writerow(result_to_row(result))
                        fh.flush()
    finally:
        if fh:
            fh.close()

    if code_graph is not None and hasattr(code_graph, 'close_all'):
        try:
            code_graph.close_all()
        except Exception:
            pass

    # End-of-run summary
    written = [r for r in results if r and not r.get('skipped')]
    n_exc = sum(1 for r in written if r.get('is_exception'))
    n_empty = sum(1 for r in written if not r.get('is_exception') and not r.get('assertion'))
    n_ok = len(written) - n_exc - n_empty
    resumed = len(done_names)
    suffix = f"  (+{resumed} resumed)" if resumed else ""
    dest = f"  → {output_file}" if output_file else ""
    print(f"[summary] written={len(written)}  assertion_ok={n_ok}  exception={n_exc}  empty={n_empty}{suffix}{dest}")

    return results


def log_exception_sr(results):
    processed = [r for r in results if not r.get('skipped')]
    exc_gt = [r for r in processed if r.get('gt_output') == 'exception']
    if not exc_gt:
        return

    fn = sum(1 for r in exc_gt if not r.get('is_exception', False))
    t_exc = len(exc_gt)
    exc_sr = (t_exc - fn) / t_exc

    print(f"\n[Exception SR] {exc_sr:.4f} ({exc_sr*100:.2f}%)  "
          f"T_exc={t_exc}  FN={fn}")


TEST_PREFIX_SOURCE = "toga-reflect/artifact/RQ2/toga-model-inputs-outputs/{project}/toga_output/oracle_preds.csv"


def merge_test_prefix_from_source(output_file, project_name):
    """Update test_prefix in output_file using TOGA oracle_preds."""
    base = Path(__file__).resolve().parent.parent
    path = base / TEST_PREFIX_SOURCE.replace("{project}", project_name)
    if not path.exists():
        logger.warning("test_prefix source not found: %s", path)
        return
    toga_df = pd.read_csv(path)
    if 'test_name' not in toga_df.columns or 'test_prefix' not in toga_df.columns:
        logger.warning("Source missing test_name/test_prefix columns")
        return
    prefix_map = dict(zip(toga_df['test_name'], toga_df['test_prefix']))
    df = pd.read_csv(output_file)
    df['test_prefix'] = df['test_name'].map(prefix_map).fillna(df['test_prefix'])
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)


def check_api(llm):
    """Startup check: verify API connectivity and tool calling support."""
    from langchain_core.tools import tool

    @tool
    def ping(x: str = '') -> str:
        """ping"""
        return 'pong'

    try:
        llm.invoke([HumanMessage(content='hi')])
    except Exception as e:
        print(f"[ERROR] API connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        llm.bind_tools([ping]).invoke([HumanMessage(content='hi')])
    except Exception as e:
        print(f"[WARN] Tool calling not supported: {e}", file=sys.stderr)
        print("[WARN] Agents will run without tool-augmented search.", file=sys.stderr)


def main():
    import warnings
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description='Solution 4: Multi-Agent Assertion Generation (Tree-sitter + KùzuDB)')
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--language', type=str, default='java',
                        choices=['java', 'python', 'javascript'],
                        help='Programming language of the project')
    parser.add_argument('--input_dir', type=str, default=os.getenv('INPUT_DIR'))
    parser.add_argument('--output_dir', type=str, default=os.getenv('OUTPUT_DIR'))
    parser.add_argument('--api_endpoint', type=str, default=os.getenv('API_ENDPOINT'))
    parser.add_argument('--model_name', type=str, default=os.getenv('MODEL_NAME'))
    parser.add_argument('--api_key', type=str, default=os.getenv('API_KEY', 'EMPTY'))
    parser.add_argument('--max_workers', type=int, default=int(os.getenv('MAX_WORKERS', '8')))
    parser.add_argument('--temperature', type=float, default=float(os.getenv('TEMPERATURE', '0.0')))
    parser.add_argument('--max_tokens', type=int, default=int(os.getenv('MAX_TOKENS', '4096')))
    parser.add_argument(
        '--streaming',
        action=argparse.BooleanOptionalAction,
        default=str(os.getenv('STREAMING', 'true')).lower() in {'1', 'true', 'yes', 'on'},
        help='Enable LLM streaming mode (default: true)',
    )
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0,
                        help='Skip first N items after done-filtering (use with --output_file to split across terminals)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV filename override (default: oracle_preds_qwen3-coder-next.csv)')
    parser.add_argument('--force_reindex', action='store_true',
                        help='Force re-parsing the project (delete existing graph DB)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    project_output = output_dir / args.project
    project_output.mkdir(parents=True, exist_ok=True)

    # Phase A: Parse project with tree-sitter and build KùzuDB graph
    project_source = str(input_dir / args.project)
    store = CodeGraph(
        project_source,
        language=args.language,
        force_reindex=args.force_reindex,
    )

    # Load test cases
    meta_llm_path = str(input_dir / args.project / 'infer_input' / 'meta_llm.csv')
    inputs_df = pd.read_csv(str(input_dir / args.project / 'infer_input' / 'inputs.csv'))
    meta_df = pd.read_csv(meta_llm_path)

    if 'test_name' in inputs_df.columns and 'test_name' in meta_df.columns:
        merged_df = inputs_df.merge(meta_df, on='test_name', how='inner', suffixes=('', '_meta'))
    else:
        # inputs.csv is generated without test_name — positional alignment by row order.
        # Both files are always produced together so row order is guaranteed to match.
        if len(inputs_df) != len(meta_df):
            raise ValueError(
                f"inputs.csv ({len(inputs_df)} rows) and meta_llm.csv ({len(meta_df)} rows) "
                "have different row counts and no 'test_name' column to join on. "
                "Cannot safely merge."
            )
        extra_cols = [c for c in meta_df.columns if c not in inputs_df.columns]
        merged_df = pd.concat(
            [inputs_df.reset_index(drop=True),
             meta_df[extra_cols].reset_index(drop=True)],
            axis=1,
        )

    items = []
    for _, row in merged_df.iterrows():
        focal_method = str(row['focal_method'])
        docstring = str(row.get('docstring', '')) if pd.notna(row.get('docstring')) else ''
        test_prefix = str(row['test_prefix']) if pd.notna(row.get('test_prefix')) else ''
        gt_output = str(row.get('GT_output', '')) if pd.notna(row.get('GT_output')) else ''
        file_path = str(row.get('file_path', '')) if pd.notna(row.get('file_path')) else ''
        test_name = str(row.get('test_name', '')) if pd.notna(row.get('test_name')) else ''
        return_type = extract_return_type(focal_method) or ''

        items.append({
            'focal_method': focal_method,
            'focal_class': extract_focal_class(test_name, args.language),
            'language': args.language,
            'docstring': docstring,
            'test_prefix': test_prefix,
            'return_type': return_type,
            'test_name': test_name,
            'file_path': file_path,
            'gt_output': gt_output
        })
    llm = ChatOpenAI(
        base_url=f"{args.api_endpoint}",
        api_key=args.api_key,
        model=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=1,
        timeout=120,
        http_client=httpx.Client(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)),
        http_async_client=httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)),
        streaming=args.streaming,
    )

    check_api(llm)

    output_file = str(project_output / (args.output_file or 'oracle_preds_qwen3-coder-next.csv'))

    results = run_inference(llm, store, args.language, items, args.max_workers,
                            output_file=output_file, offset=args.offset, limit=args.limit)

    if args.language == 'java':
        merge_test_prefix_from_source(output_file, args.project)
    log_exception_sr(results)


if __name__ == '__main__':
    main()

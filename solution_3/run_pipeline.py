import os
import re
import sys
import csv
import logging
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

logger = logging.getLogger(__name__)

from method_store import MethodStore
from graph import build_graph


def post_process_assertion(raw_assertion):
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
    for line in raw_assertion.split('\n'):
        line = line.strip()
        if line.startswith('assert'):
            raw_assertion = line
            break

    raw_assertion = _strip_trailing_comment(raw_assertion)
    return fix_assertion(raw_assertion)


def _strip_trailing_comment(line):
    """Remove trailing Java // comment, but only when // is outside a string literal."""
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
        if not in_str and ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
            return line[:i].rstrip()
    return line


_NEG_LITERAL = re.compile(r'^-\d+(?:\.\d+)?[LlFfDd]?$')


def _wrap_negative_literals(assertion):
    """Wrap bare negative number literals in parentheses: -2L -> (-2L)."""
    match = re.match(r'(assert\w+\()(.+)(\);)$', assertion.rstrip())
    if not match:
        return assertion

    prefix, inner, suffix = match.group(1), match.group(2), match.group(3)
    args = _split_args(inner)
    changed = False
    for i, arg in enumerate(args):
        stripped = arg.strip()
        if _NEG_LITERAL.match(stripped):
            args[i] = f'({stripped})'
            changed = True
        else:
            args[i] = stripped
    if not changed:
        return assertion
    return prefix + ', '.join(args) + suffix


def _split_args(s):
    """Split assertion arguments respecting parentheses depth."""
    args, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if ch in '({':
            depth += 1
        elif ch in ')}':
            depth -= 1
        elif ch == ',' and depth == 0:
            args.append(s[start:i])
            start = i + 1
    args.append(s[start:])
    return args


def _count_parens(s):
    """Count '(' and ')' outside of string literals (E2 guard)."""
    open_p = close_p = 0
    in_str = escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == '(':
                open_p += 1
            elif ch == ')':
                close_p += 1
    return open_p, close_p


def fix_assertion(assertion):
    if not assertion:
        return assertion

    assertion = assertion.strip()

    # E2: count parens outside string literals to avoid being misled by '(' inside strings
    open_p, close_p = _count_parens(assertion)
    open_b = assertion.count('{')
    close_b = assertion.count('}')

    if '() -> {' in assertion and close_b < open_b:
        missing_b = open_b - close_b
        assertion = assertion.rstrip(';') + '}' * missing_b
        open_p, close_p = _count_parens(assertion)
        if close_p < open_p:
            assertion += ')' * (open_p - close_p)
    elif close_p < open_p:
        assertion = assertion.rstrip(';') + ')' * (open_p - close_p)

    if not assertion.endswith(';'):
        assertion += ';'

    assertion = _wrap_negative_literals(assertion)

    return assertion


def extract_focal_class(test_name):
    """Extract the class under test from test_name.
    e.g. 'org...ContentWriteProgress_ESTest::test1' -> 'ContentWriteProgress'
    """
    cls = test_name.split('::')[0].rsplit('.', 1)[-1]
    for suffix in ('_ESTest_scaffolding', '_ESTest'):
        if cls.endswith(suffix):
            return cls[:-len(suffix)]
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


def process_single(compiled_graph, item):
    initial_state = {
        'focal_method': item['focal_method'],
        'focal_class': item.get('focal_class', ''),
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
                "tags": ["solution3"],
            },
        )
        is_exception = result.get('is_exception', False)
        assertion = 'exception' if is_exception else post_process_assertion(result.get('assertion', ''))
        return {
            **item,
            'assertion': assertion,
            'is_exception': is_exception,
        }
    except Exception as e:
        logger.warning("Graph invoke failed for %s: %s", item.get('test_name', '?'), e)
        return {**item, 'assertion': '', 'is_exception': False}


def run_inference(compiled_graph, items, max_workers):
    results: list[dict | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, compiled_graph, item): i
            for i, item in enumerate(items)
        }
        for future in tqdm(as_completed(futures), total=len(items), desc="Inference"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning("Future failed for item %d: %s", idx, e)
                results[idx] = {**items[idx], 'assertion': ''}
    return results


def save_csv(results, output_file, quoting=csv.QUOTE_ALL):
    rows = [{
        'test_name': r.get('test_name', ''),
        'test_prefix': r.get('test_prefix', ''),
        'file_path': r.get('file_path', ''),
        'assert_pred': '' if r.get('is_exception', False) else r.get('assertion', ''),
    } for r in results]
    pd.DataFrame(rows).to_csv(output_file, index=False, quoting=quoting)



def log_exception_sr(results):
    exc_gt = [r for r in results if r.get('gt_output') == 'exception']
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


def _check_api(llm):
    """Startup check: verify API connectivity and tool calling support."""
    from langchain_core.tools import tool as _tool

    @_tool
    def _ping(x: str = '') -> str:
        """ping"""
        return 'pong'

    try:
        llm.invoke([HumanMessage(content='hi')])
    except Exception as e:
        print(f"[ERROR] API connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        llm.bind_tools([_ping]).invoke([HumanMessage(content='hi')])
    except Exception as e:
        print(f"[WARN] Tool calling not supported: {e}", file=sys.stderr)
        print("[WARN] Agents will run without tool-augmented search.", file=sys.stderr)


def main():
    import warnings
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description='Solution 3: Multi-Agent Assertion Generation')
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default=os.getenv('INPUT_DIR'))
    parser.add_argument('--output_dir', type=str, default=os.getenv('OUTPUT_DIR'))
    parser.add_argument('--api_endpoint', type=str, default=os.getenv('API_ENDPOINT'))
    parser.add_argument('--model_name', type=str, default=os.getenv('MODEL_NAME'))
    parser.add_argument('--api_key', type=str, default=os.getenv('API_KEY', 'EMPTY'))
    parser.add_argument('--max_workers', type=int, default=int(os.getenv('MAX_WORKERS', '8')))
    parser.add_argument('--temperature', type=float, default=float(os.getenv('TEMPERATURE', '0.0')))
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    project_output = output_dir / args.project
    project_output.mkdir(parents=True, exist_ok=True)

    # Phase A: Parse project AST directly from source
    project_source = str(input_dir / args.project)
    store = MethodStore(project_source)

    # Load test cases
    meta_llm_path = str(input_dir / args.project / 'infer_input' / 'meta_llm.csv')
    inputs_df = pd.read_csv(str(input_dir / args.project / 'infer_input' / 'inputs.csv'))
    meta_df = pd.read_csv(meta_llm_path)

    items = []
    for idx in range(len(inputs_df)):
        input_row, meta_row = inputs_df.iloc[idx], meta_df.iloc[idx]
        focal_method = str(input_row['focal_method'])
        docstring = str(input_row.get('docstring', '')) if pd.notna(input_row.get('docstring')) else ''
        test_prefix = str(meta_row['test_prefix']) if pd.notna(meta_row.get('test_prefix')) else ''
        gt_output = str(meta_row.get('GT_output', '')) if pd.notna(meta_row.get('GT_output')) else ''
        file_path = str(meta_row.get('file_path', '')) if pd.notna(meta_row.get('file_path')) else ''
        test_name = str(meta_row.get('test_name', '')) if pd.notna(meta_row.get('test_name')) else ''
        return_type = extract_return_type(focal_method) or ''

        items.append({
            'focal_method': focal_method,
            'focal_class': extract_focal_class(test_name),
            'docstring': docstring,
            'test_prefix': test_prefix,
            'return_type': return_type,
            'test_name': test_name,
            'file_path': file_path,
            'gt_output': gt_output,s
        })
    if args.limit is not None:
        items = items[:args.limit]

    llm = ChatOpenAI(
        base_url=f"{args.api_endpoint}",
        api_key=args.api_key,
        model=args.model_name,
        temperature=args.temperature,
        max_retries=5,
        timeout=600,
    )

    _check_api(llm)
    compiled_graph = build_graph(llm, store)
    results = run_inference(compiled_graph, items, args.max_workers)

    model_short = args.model_name.split('/')[-1].replace('.', '-')
    output_file = str(project_output / f'oracle_preds_{model_short}.csv')
    save_csv(results, output_file)
    merge_test_prefix_from_source(output_file, args.project)
    log_exception_sr(results)


if __name__ == '__main__':
    main()

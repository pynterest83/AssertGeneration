import argparse
import re
import pandas as pd
from pathlib import Path

FAIL_CATCH_RE = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE | re.DOTALL)

PROJECTS = [
    "Cli", "Codec", "Compress", "Csv", "Gson",
    "JacksonCore", "JacksonDatabind", "Jsoup", "JxPath", "Lang", "Math",
]

# mapping test_name to (project, bug_num) from solution output to toga d4j format
def load_meta_map(evo_meta_path: Path) -> dict:
    meta = pd.read_csv(evo_meta_path)
    mapping = {}
    for _, row in meta.iterrows():
        tn = row['test_name']
        if tn not in mapping:
            mapping[tn] = (row['project'], int(row['bug_num']))
    return mapping


def is_exception_prefix(test_prefix: str) -> bool:
    return bool(re.search(FAIL_CATCH_RE, str(test_prefix)))


def infer_except_pred(assert_pred) -> int:
    """Return 1 if solution_3 predicted exception (empty/NaN assert_pred), else 0."""
    if pd.isna(assert_pred):
        return 1
    s = str(assert_pred).strip()
    return 1 if s == "" else 0


def collect_predictions(output_dir: Path, model_short: str) -> pd.DataFrame:
    """Concatenate oracle_preds_{model_short}.csv across all projects."""
    all_rows = []
    for project in PROJECTS:
        pred_file = output_dir / project / f"oracle_preds_{model_short}.csv"
        if not pred_file.exists():
            print(f"  [SKIP] {pred_file} not found")
            continue
        df = pd.read_csv(pred_file)
        all_rows.append(df)
        print(f"  [OK]   {project}: {len(df)} rows")

    if not all_rows:
        raise FileNotFoundError(f"No oracle_preds_{model_short}.csv files found in {output_dir}")
    return pd.concat(all_rows, ignore_index=True)


def convert(output_dir: Path, input_dir: Path, evo_meta_path: Path, model_short: str):
    meta_map = load_meta_map(evo_meta_path)

    preds = collect_predictions(output_dir, model_short)
    print(f"Total rows: {len(preds)}")

    rows = []
    missing = []
    for idx, row in preds.iterrows():
        tn = row['test_name']
        if tn not in meta_map:
            missing.append(tn)
            continue
        project, bug_num = meta_map[tn]

        # test_prefix in solution_3 output = clean version
        test_prefix = str(row.get('test_prefix', '')) if pd.notna(row.get('test_prefix')) else ''

        assert_pred = row.get('assert_pred', '')
        except_pred = infer_except_pred(assert_pred)

        rows.append({
            'id': idx + 1,
            'project': project,
            'bug_num': bug_num,
            'test_name': tn,
            'test_prefix': test_prefix,
            'except_pred': except_pred,
            'assert_pred': assert_pred if not except_pred else float('nan'),
        })

    if missing:
        print(f"[WARN] {len(missing)} test_names not found in meta: {missing[:5]}")

    result = pd.DataFrame(rows)

    # Write combined oracle_preds.csv
    bug_dir = output_dir / 'bug_detection'
    bug_dir.mkdir(parents=True, exist_ok=True)
    combined_path = bug_dir / 'oracle_preds.csv'
    result.to_csv(combined_path, index=False)
    print(f"\nWrote combined oracle_preds.csv → {combined_path}  ({len(result)} rows)")

    # Split into three sub-types to run for result
    _split_preds(result, bug_dir)


def _split_preds(df: pd.DataFrame, bug_dir: Path):
    exc_rows, assert_rows, prefix_rows = [], [], []

    for _, row in df.iterrows():
        tp = str(row['test_prefix'])
        has_exc_pattern = is_exception_prefix(tp)
        pred_is_nan = pd.isna(row['assert_pred']) or str(row['assert_pred']).strip() == ''

        if has_exc_pattern:
            exc_rows.append(row)
        elif not pred_is_nan:
            assert_rows.append(row)
        else:
            prefix_rows.append(row)

    for sub_dir, subset, label in [
        ('assertion_prefix', assert_rows, 'assertion_prefix'),
        ('exception_prefix', exc_rows, 'exception_prefix'),
        ('prefix_only', prefix_rows, 'prefix_only'),
    ]:
        out_dir = bug_dir / sub_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(subset)
        out_path = out_dir / 'oracle_preds.csv'
        out_df.to_csv(out_path, index=False)
        print(f"  [{label}] {len(out_df)} rows {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert solution_3 output to TOGLL bug-detection format')
    parser.add_argument('--output_dir', type=str, default='data/RQ3/output')
    parser.add_argument('--input_dir', type=str, default='data/RQ3/input')
    parser.add_argument('--evo_meta', type=str,
                        default='togll/RQ5/TOGLL_prediction/evosuite_reaching_tests/meta.csv')
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent.parent

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else base / p

    output_dir = resolve(args.output_dir)
    input_dir = resolve(args.input_dir)
    evo_meta = resolve(args.evo_meta)
    model_short = args.model_name.split('/')[-1].replace('.', '-')

    convert(output_dir, input_dir, evo_meta, model_short)


if __name__ == '__main__':
    main()

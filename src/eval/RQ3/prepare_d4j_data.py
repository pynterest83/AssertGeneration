import argparse
import re
import pandas as pd
from pathlib import Path

# căn lại đầu dòng, tab
def clean_data(data):
    lines = data.strip().split('\n')
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 2:
        return '\n'.join(non_empty)
    cleaned = ([non_empty[0].strip()]
               + ['\t' + l.lstrip() for l in non_empty[1:-1]]
               + [non_empty[-1].strip()])
    return '\n'.join(cleaned)

# xóa bỏ dòng trống
def remove_empty_line(data):
    lines = data.strip().split('\n')
    return '\n'.join(l for l in lines if l.strip())

def clean_test_prefix(test_prefix):
    # Remove '// Undeclared exception!' comment lines
    pattern = r'^.*\/\/\s*Undeclared exception!.*$'
    lines = test_prefix.split('\n')
    filtered = [l for l in lines if not re.match(pattern, l) and l.strip()]
    tp = '\n'.join(filtered)
    # Remove try/catch/fail blocks
    if "try" in tp and "catch" in tp:
        tp = re.sub(r'try\s*\{', '', tp)
        tp = re.sub(r'fail\s*\([^)]*\);', '', tp)
        tp = re.sub(r'\}\s*catch\s*\([^)]*\)\s*\{.*?\}', '', tp, flags=re.DOTALL)
        tp = clean_data(tp)
    else:
        # Remove bare assertion lines if any
        if "assert" in tp:
            assert_re = re.compile(r'assert\w*\(.*\);')
            tp = re.sub(assert_re, '', tp)
            tp = remove_empty_line(tp)

    return tp

def derive_file_path(test_name):
    class_part = test_name.split('::')[0]
    return class_part.replace('.', '/') + '.java'

def row_gt_output(row):
    if row['exception_lbl'] is True or str(row['exception_lbl']).lower() == 'true':
        return "exception"
    val = row['assertion_lbl']
    if pd.notna(val) and str(val).strip():
        return str(val).strip()
    return ""

def process(evo_dir, output_dir):
    inputs_path = evo_dir / 'inputs.csv'
    meta_path = evo_dir / 'meta.csv'

    inputs_df = pd.read_csv(inputs_path)
    meta_df = pd.read_csv(meta_path)

    assert len(inputs_df) == len(meta_df), (
        f"Row count mismatch: inputs={len(inputs_df)}, meta={len(meta_df)}")

    # Combine into a single df
    # keep all 374 rows, same as TOGLL's approach.

    combined = meta_df.copy()
    combined['focal_method'] = inputs_df['focal_method'].values
    combined['raw_test_prefix'] = inputs_df['test_prefix'].values
    combined['docstring'] = inputs_df['docstring'].values

    projects = combined['project'].unique()
    global_id = 1

    for project in sorted(projects):
        proj_df = combined[combined['project'] == project].copy()

        all_rows = []
        for _, row in proj_df.iterrows():
            tn = row['test_name']
            raw_tp = str(row['raw_test_prefix']) if pd.notna(row['raw_test_prefix']) else ""
            clean_tp = clean_test_prefix(raw_tp)
            focal = str(row['focal_method']) if pd.notna(row['focal_method']) else ""
            doc = str(row['docstring']) if pd.notna(row['docstring']) else ""

            all_rows.append({
                'id': global_id,
                'project': project,
                'bug_num': row['bug_num'],
                'test_name': tn,
                'focal_method': focal,
                'raw_test_prefix': raw_tp,
                'clean_test_prefix': clean_tp,
                'docstring': doc,
                'GT_output': row_gt_output(row),
                'file_path': derive_file_path(tn),
            })
            global_id += 1

        all_df = pd.DataFrame(all_rows)

        infer_dir = output_dir / project / 'infer_input'
        infer_dir.mkdir(parents=True, exist_ok=True)

        # inputs.csv: what solution_3 agents see (focal_method, raw test_prefix, docstring)
        inputs_out = all_df[['focal_method', 'raw_test_prefix', 'docstring']].rename(
            columns={'raw_test_prefix': 'test_prefix'})
        inputs_out.to_csv(infer_dir / 'inputs.csv', index=False)

        # meta_llm.csv: solution_3 metadata
        meta_out = all_df[['id', 'file_path', 'test_name', 'raw_test_prefix', 'GT_output']].rename(
            columns={'raw_test_prefix': 'test_prefix'})
        meta_out.to_csv(infer_dir / 'meta_llm.csv', index=False)

        unique_tn = all_df['test_name'].nunique()
        print(f"  [{project}] {len(all_df)} rows ({unique_tn} unique test_names) → {infer_dir}")

    print(f"\nDone. Total rows: {global_id - 1}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evo_dir', type=str, default='togll/RQ5/TOGLL_prediction/evosuite_reaching_tests')
    parser.add_argument('--output_dir', type=str, default='data/RQ3/input')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    evo_dir = Path(args.evo_dir) if Path(args.evo_dir).is_absolute() else base / args.evo_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else base / args.output_dir

    process(evo_dir, output_dir)

if __name__ == "__main__":
    main()
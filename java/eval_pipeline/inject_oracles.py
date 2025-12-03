"""
Inject RQ2 oracle predictions into RQ4 test files
"""

import json
import re
from pathlib import Path
import pandas as pd


def inject_oracle_into_test(test_content, test_name, oracle):
    """Replace oracle in test method"""
    pattern = rf'(@Test[^\n]*\n\s*public void {re.escape(test_name)}[^{{]*\{{[^}}]*?)(\}})'
    
    def replace_fn(match):
        test_body = match.group(1)
        test_body = re.sub(r'assert\w*\([^;]*\);', '', test_body)
        test_body = re.sub(r'fail\([^)]*\);', '', test_body)
        
        if oracle and oracle != 'exception':
            if not oracle.endswith(';'):
                oracle_stmt = oracle + ';'
            else:
                oracle_stmt = oracle
            return test_body + '\n' + oracle_stmt + '\n}'
        return test_body + '\n}'
    
    return re.sub(pattern, replace_fn, test_content, flags=re.DOTALL)


def inject_project_oracles(rq2_results_dir, rq4_project_dir, project_name):
    """Inject oracles for one project"""
    predictions_file = rq2_results_dir / project_name / 'predictions.json'
    if not predictions_file.exists():
        return
    
    predictions = json.load(open(predictions_file))
    
    rq2_meta = Path(__file__).parent.parent / f'RQ2/inference/inference_data/{project_name}/meta_llm.csv'
    if not rq2_meta.exists():
        return
    
    meta_df = pd.read_csv(rq2_meta)
    
    for pred in predictions:
        pred_id = pred['id']
        oracle = pred['prediction']
        
        meta_row = meta_df[meta_df['id'] == pred_id]
        if meta_row.empty:
            continue
        
        file_path = meta_row.iloc[0]['file_path']
        test_name = meta_row.iloc[0]['test_name'].split('::')[1]
        
        target_file = rq4_project_dir / 'llm_oracle' / 'test' / 'java' / file_path.replace('/', '\\').split('test\\java\\')[1]
        
        if not target_file.exists():
            continue
        
        content = target_file.read_text(encoding='utf-8')
        new_content = inject_oracle_into_test(content, test_name, oracle)
        
        if new_content != content:
            target_file.write_text(new_content, encoding='utf-8')


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq2-results', type=str, required=True)
    parser.add_argument('--rq4-artifacts', type=str, required=True)
    parser.add_argument('--projects', nargs='+')
    
    args = parser.parse_args()
    
    rq2_results = Path(args.rq2_results)
    rq4_artifacts = Path(args.rq4_artifacts)
    
    if args.projects:
        projects = args.projects
    else:
        projects = [p.name for p in rq2_results.iterdir() if p.is_dir() and (p / 'predictions.json').exists()]
    
    for project in projects:
        print(f'Injecting {project}...')
        rq4_project = rq4_artifacts / project / 'client'
        if rq4_project.exists():
            inject_project_oracles(rq2_results, rq4_project, project)
        print(f'Done: {project}')


if __name__ == '__main__':
    main()


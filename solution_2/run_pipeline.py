import re
import json
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from extract_project_elements import ExtractProjectElements
from vector_project_elements import VectorProjectElements
from prompt_builder import RUEPromptBuilder
from utils.api_inference import APIInference


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_prediction(prediction):
    if not prediction:
        return prediction
    prediction = re.sub(r'```\w*\n?', '', prediction)
    prediction = re.sub(r'```', '', prediction)
    lines = prediction.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('assert') or line.startswith('fail'):
            return fix_assertion(line)
    return fix_assertion(lines[0].strip()) if lines else prediction


def fix_assertion(assertion):
    if not assertion:
        return assertion
    open_p, close_p = assertion.count('('), assertion.count(')')
    open_b, close_b = assertion.count('{'), assertion.count('}')
    if '() -> {' in assertion and close_b < open_b:
        missing_b = open_b - close_b
        missing_p = open_p - close_p - missing_b
        assertion = assertion.rstrip(';') + '}' * missing_b + ')' * max(0, missing_p + missing_b)
    elif close_p < open_p:
        assertion = assertion.rstrip(';') + ')' * (open_p - close_p)
    if not assertion.endswith(';'):
        assertion += ';'
    return assertion


def process_item(client, item, max_new_tokens, temperature):
    if item.get('gt_output', '') == 'exception':
        return {**item, 'prediction': 'exception'}
    prediction = client.generate(prompt=item['prompt'], max_new_tokens=max_new_tokens, temperature=temperature)
    return {**item, 'prediction': clean_prediction(prediction)}


def run_inference(client, prompts, max_new_tokens, temperature, max_workers):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, client, item, max_new_tokens, temperature): i for i, item in enumerate(prompts)}
        for future in tqdm(as_completed(futures), total=len(prompts), desc="Inference"):
            results[futures[future]] = future.result()
    return results


def save_csv(results, output_file):
    rows = [{
        'test_name': r.get('test_name', ''),
        'test_prefix': r.get('test_prefix', ''),
        'file_path': r.get('file_path', ''),
        'assert_pred': '' if r.get('gt_output') == 'exception' else r.get('prediction', '')
    } for r in results]
    pd.DataFrame(rows).to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--api_endpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--returntype', action='store_true')
    parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--skip_vector', action='store_true')
    parser.add_argument('--skip_prompts', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    project_output = output_dir / args.project
    project_output.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        ExtractProjectElements(str(input_dir), str(output_dir)).extract_project(args.project)

    if not args.skip_vector and args.external:
        VectorProjectElements().vectorize_project(str(output_dir), args.project)

    features = []
    if args.external:
        features.append('external')
    if args.returntype:
        features.append('returntype')
    feature_suffix = '_'.join(features) if features else 'baseline'

    if not args.skip_prompts:
        RUEPromptBuilder().process_project(args.project, str(input_dir), str(output_dir),
                                           use_external=args.external, use_returntype=args.returntype)

    prompts = load_jsonl(str(project_output / f'prompts_{feature_suffix}_bow.jsonl'))
    client = APIInference(api_endpoint=args.api_endpoint, api_key=args.api_key, model_name=args.model_name)
    results = run_inference(client, prompts, 100, 0.0, 8)
    
    # Extract model short name for filename
    model_short = args.model_name.split('/')[-1].replace('.', '-')
    save_csv(results, str(project_output / f'oracle_preds_{feature_suffix}_{model_short}.csv'))


if __name__ == '__main__':
    main()

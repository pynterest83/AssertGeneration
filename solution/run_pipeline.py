import os
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
    parser.add_argument('--api_endpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--embedding', type=str, choices=['bow', 'semantic'], default='bow')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--projects_base_dir', type=str, default='../RQ2/EvoSuiteTests')
    parser.add_argument('--rq2_data', type=str, default='../RQ2/inference/inference_data')
    parser.add_argument('--results_dir', type=str, default='../results')
    parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--skip_vector', action='store_true')
    parser.add_argument('--skip_prompts', action='store_true')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    project_dir = results_dir / args.project

    # Step 1: Extract
    if not args.skip_extract:
        extractor = ExtractProjectElements(args.projects_base_dir, str(results_dir))
        extractor.extract_project(args.project)

    # Step 2: Vectorize
    if not args.skip_vector:
        vectorizer = VectorProjectElements(embedding_type=args.embedding)
        vectorizer.vectorize_project(str(results_dir), args.project)

    # Step 3: Build prompts
    if not args.skip_prompts:
        builder = RUEPromptBuilder(embedding_type=args.embedding)
        builder.process_project(args.project, args.rq2_data, str(results_dir))

    # Step 4: Inference
    prompt_file = project_dir / f'prompts_external_{args.embedding}.jsonl'
    prompts = load_jsonl(str(prompt_file))

    client = APIInference(api_endpoint=args.api_endpoint, api_key=args.api_key, model_name=args.model_name)
    results = run_inference(client, prompts, args.max_new_tokens, args.temperature, args.max_workers)

    # Step 5: Save
    output_dir = project_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)
    save_csv(results, str(output_dir / 'oracle_preds.csv'))


if __name__ == '__main__':
    main()

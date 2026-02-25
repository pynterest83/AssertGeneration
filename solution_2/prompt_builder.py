import os
import re
import json
import pandas as pd
from tqdm import tqdm

from utils.bow_embedding import BagOfWordsEmbedding
from utils.semantic_embedding import SemanticEmbedding
from prompt_instruction_template import build_prompt_with_instruction

TEST_PATTERNS = ['_ESTest', 'Test', '_test', 'test_']
ASSERT_PATTERNS = ['assertEquals', 'assertTrue', 'assertFalse', 'assertNull', 
                   'assertNotNull', 'assertSame', 'assertNotSame', 'assertThrows',
                   'assertThat', 'assertArrayEquals', 'fail(']


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_method_name(code):
    match = re.search(r'(?:public|private|protected|static|\s)+\s*(?:<[^>]+>\s*)?\w+(?:<[^>]+>)?\s+(\w+)\s*\(', code)
    return match.group(1) if match else None


def extract_return_type(code):
    match = re.search(r'(?:public|private|protected|static|\s)+\s*(?:<[^>]+>\s*)?(\w+(?:<[^>]+>)?)\s+\w+\s*\(', code)
    return match.group(1) if match else None


def is_test_method(method):
    meta = method['metadata']
    class_name = meta.get('class', '')
    method_name = meta.get('name', '')
    fpath_str = '/'.join(meta.get('fpath_tuple', []))
    body = meta.get('body_raw', '')
    
    if any(p in class_name for p in TEST_PATTERNS):
        return True
    if method_name.startswith('test'):
        return True
    if 'test' in fpath_str.lower() and 'evosuite' in fpath_str.lower():
        return True
    if any(p in body for p in ASSERT_PATTERNS):
        return True
    return False


class RUEPromptBuilder:
    def __init__(self, embedding_type='bow', model_name='unixcoder'):
        self.embedding_type = embedding_type
        self.embedder = BagOfWordsEmbedding() if embedding_type == 'bow' else SemanticEmbedding(model_name=model_name)
    
    def find_external_usages(self, focal_method_name, methods):
        return [m for m in methods 
                if not is_test_method(m) 
                and focal_method_name in m['metadata'].get('body_raw', '')
                and m['metadata']['name'] != focal_method_name]
    
    def rank_by_similarity(self, query, candidates, top_k=10):
        if not candidates:
            return []
        query_emb = self.embedder.build(query)
        scored = []
        for cand in candidates:
            cand_emb = cand.get('embeddings', {}).get('body') or self.embedder.build(cand['metadata'].get('body_raw', ''))
            score = self.embedder.similarity(query_emb, cand_emb)
            scored.append((cand, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def build_context(self, usages, max_lines=15):
        if not usages:
            return ""
        parts = ["// Here are some methods that use the focal method:", "// " + "-" * 50]
        for method, _ in usages:
            lines = method['metadata'].get('body_raw', '').split('\n')
            if len(lines) > max_lines:
                lines = lines[:max_lines] + ['// ... (truncated)']
            parts.extend([f'// {line}' for line in lines])
            parts.append("// " + "-" * 50)
        return '\n'.join(parts)
    
    def process_project(self, project_name, input_dir, output_dir, top_k=5, use_external=False, use_returntype=False):
        inputs_df = pd.read_csv(os.path.join(input_dir, project_name, 'infer_input', 'inputs.csv'))
        meta_df = pd.read_csv(os.path.join(input_dir, project_name, 'infer_input', 'meta_llm.csv'))
        
        emb_suffix = 'bow' if self.embedding_type == 'bow' else self.embedder.model_name
        methods = load_jsonl(os.path.join(output_dir, project_name, f'methods_embeddings_{emb_suffix}.jsonl')) if use_external else []
        
        results = []
        for idx in tqdm(range(len(inputs_df)), desc="Building prompts"):
            input_row, meta_row = inputs_df.iloc[idx], meta_df.iloc[idx]
            
            focal_method = str(input_row['focal_method'])
            docstring = str(input_row.get('docstring', '')) if pd.notna(input_row.get('docstring')) else ''
            test_prefix = str(meta_row['test_prefix']) if pd.notna(meta_row.get('test_prefix')) else ''
            ground_truth = str(meta_row.get('GT_output', '')) if pd.notna(meta_row.get('GT_output')) else ''
            file_path = str(meta_row.get('file_path', '')) if pd.notna(meta_row.get('file_path')) else ''
            test_name = str(meta_row.get('test_name', '')) if pd.notna(meta_row.get('test_name')) else ''
            
            focal_name = extract_method_name(focal_method)
            return_type = extract_return_type(focal_method) if use_returntype else None
            
            usages_info, context = [], ""
            if use_external and focal_name:
                usages = self.find_external_usages(focal_name, methods)
                if usages:
                    ranked = self.rank_by_similarity(test_prefix, usages, top_k)
                    context = self.build_context(ranked)
                    usages_info = [{'method': u['metadata']['name'], 'class': u['metadata']['class'], 'score': s} for u, s in ranked]
            
            if return_type:
                context = f"// Return type: {return_type}\n{context}" if context else f"// Return type: {return_type}"
            
            prompt = build_prompt_with_instruction(
                focal_method=focal_method, 
                test_prefix=test_prefix, 
                docstring=docstring, 
                context=context,
                return_type=return_type if return_type else ""
            )
            
            results.append({
                'id': int(meta_row.get('id', idx)),
                'test_name': test_name,
                'file_path': file_path,
                'prompt': prompt,
                'focal_method': focal_method,
                'focal_method_name': focal_name,
                'test_prefix': test_prefix,
                'docstring': docstring,
                'gt_output': ground_truth,
                'usages': usages_info,
                'has_context': len(usages_info) > 0
            })
        
        features = []
        if use_external:
            features.append('external')
        if use_returntype:
            features.append('returntype')
        feature_suffix = '_'.join(features) if features else 'baseline'
        
        output_path = os.path.join(output_dir, project_name)
        os.makedirs(output_path, exist_ok=True)
        save_jsonl(results, os.path.join(output_path, f'prompts_{feature_suffix}_{emb_suffix}.jsonl'))
        return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--embedding', type=str, choices=['bow', 'semantic'], default='bow')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--returntype', action='store_true')
    args = parser.parse_args()
    
    builder = RUEPromptBuilder(embedding_type=args.embedding)
    builder.process_project(args.project, args.input_dir, args.output_dir, args.top_k,
                           use_external=args.external, use_returntype=args.returntype)


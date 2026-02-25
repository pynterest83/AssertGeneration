import os
import json
import argparse
from tqdm import tqdm

from utils.bow_embedding import BagOfWordsEmbedding
from utils.semantic_embedding import SemanticEmbedding, SUPPORTED_MODELS


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def build_signature(meta):
    params_str = ' '.join([p['type'] for p in meta.get('parameters', [])])
    return f"{meta.get('return_type', 'void')} {meta['name']} {params_str}"


class VectorProjectElements:
    def __init__(self, embedding_type='bow', model_name='unixcoder', device=None):
        self.embedding_type = embedding_type
        self.embedder = BagOfWordsEmbedding() if embedding_type == 'bow' else SemanticEmbedding(model_name=model_name, device=device)
    
    def vectorize_project(self, input_dir, project_name):
        methods = load_jsonl(os.path.join(input_dir, project_name, 'methods.jsonl'))
        
        signatures = [build_signature(m['metadata']) for m in methods]
        bodies = [m['metadata'].get('body_raw', '') for m in methods]
        names = [m['metadata']['name'] for m in methods]
        
        sig_embeddings = self.embedder.build_batch(signatures)
        body_embeddings = self.embedder.build_batch(bodies)
        name_embeddings = self.embedder.build_batch(names)
        
        vectorized = []
        for i, method in enumerate(methods):
            vectorized.append({
                'metadata': {**method['metadata'], 'signature': signatures[i]},
                'embeddings': {
                    'signature': sig_embeddings[i],
                    'body': body_embeddings[i],
                    'name': name_embeddings[i]
                }
            })
        
        output_path = os.path.join(input_dir, project_name, f'methods_embeddings_{self.embedder}.jsonl')
        save_jsonl(vectorized, output_path)
        return vectorized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default='../results')
    parser.add_argument('--embedding', type=str, choices=['bow', 'semantic'], default='bow')
    parser.add_argument('--model', type=str, default='unixcoder', choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    vectorizer = VectorProjectElements(embedding_type=args.embedding, model_name=args.model, device=args.device)
    vectorizer.vectorize_project(args.input_dir, args.project)

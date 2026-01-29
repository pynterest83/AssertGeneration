import torch
import numpy as np
from typing import List, Union

SUPPORTED_MODELS = {
    'unixcoder': 'microsoft/unixcoder-base',
    'codebert': 'microsoft/codebert-base',
    'graphcodebert': 'microsoft/graphcodebert-base',
    'codet5': 'Salesforce/codet5-base',
    'starencoder': 'bigcode/starencoder',
}


class SemanticEmbedding:
    def __init__(self, model_name='unixcoder', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model_name = model_name
        self.model_id = SUPPORTED_MODELS.get(model_name, model_name)
        
        self._load_model()
    
    def _load_model(self):
        if self.model_name == 'unixcoder':
            self._load_unixcoder()
        else:
            self._load_huggingface()
    
    def _load_unixcoder(self):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.embed_type = 'unixcoder'
    
    def _load_huggingface(self):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.embed_type = 'huggingface'
    
    def build(self, text, max_length=512) -> List[float]:
        if not text:
            return [0.0] * 768
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding.squeeze().cpu().tolist()
    
    def build_batch(self, texts, max_length=512, batch_size=32) -> List[List[float]]:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_texts = [t if t else "" for t in batch_texts]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
            
            embeddings.extend(batch_embeddings.cpu().tolist())
        
        return embeddings
    
    def similarity(self, emb1, emb2) -> float:
        if isinstance(emb1, list):
            emb1 = torch.tensor(emb1)
        if isinstance(emb2, list):
            emb2 = torch.tensor(emb2)
        
        return torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()
    
    def similarity_batch(self, query_emb, doc_embs) -> List[float]:
        if isinstance(query_emb, list):
            query_emb = torch.tensor(query_emb)
        if isinstance(doc_embs, list):
            doc_embs = torch.tensor(doc_embs)
        
        query_emb = query_emb.unsqueeze(0)
        scores = torch.nn.functional.cosine_similarity(query_emb, doc_embs, dim=1)
        return scores.tolist()
    
    def __str__(self):
        return self.model_name


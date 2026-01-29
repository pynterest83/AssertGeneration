import numpy as np
import tiktoken


class BagOfWordsEmbedding:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")
    
    def tokenize(self, text):
        if not text:
            return []
        return self.tokenizer.encode_ordinary(text)
    
    def build(self, text):
        return self.tokenize(text)
    
    def build_batch(self, texts):
        return [self.build(text) for text in texts]
    
    def similarity(self, tokens1, tokens2):
        if not tokens1 or not tokens2:
            return 0.0
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        return float(intersection) / union
    
    def similarity_batch(self, query_tokens, doc_tokens_list):
        return [self.similarity(query_tokens, doc_tokens) for doc_tokens in doc_tokens_list]
    
    def __str__(self):
        return 'bow'


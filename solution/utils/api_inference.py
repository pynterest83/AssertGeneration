import os
import time
from typing import List, Optional, Dict, Any
import requests


class APIInference:
    def __init__(
        self,
        api_endpoint
        api_key
        model_name
    ):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.0,
        top_p=1.0,
        stop_tokens=None,
        **kwargs
    ):
        endpoint = f"{self.api_endpoint}/v1/chat/completions"
        
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': temperature,
        }
        
        if stop_tokens:
            payload['stop'] = stop_tokens
        
        payload.update(kwargs)
        
        response = requests.post(
            endpoint,
            headers=self.headers,
            json=payload,
        )
                
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    
    def generate_completion(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.0,
        top_p=1.0,
        stop_tokens=None,
        **kwargs
    ):
        endpoint = f"{self.api_endpoint}/v1/completions"
        
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'max_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
        }
        
        payload.update(kwargs)

        response = requests.post(
            endpoint,
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
                
        data = response.json()
        return data['choices'][0]['text'].strip()

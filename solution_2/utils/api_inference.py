import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class APIInference:
    def __init__(
        self,
        api_endpoint,
        api_key='',
        model_name=''
    ):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retry))
        self.session.mount('https://', HTTPAdapter(max_retries=retry))

    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.0,
    ):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_new_tokens,
            'temperature': temperature
        }
        
        for attempt in range(3):
            try:
                response = self.session.post(
                    f'{self.api_endpoint}/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                if response.status_code != 200:
                    time.sleep(2 ** attempt)
                    continue
                data = response.json()
                return data['choices'][0]['message']['content']
            except (requests.exceptions.JSONDecodeError, KeyError, requests.exceptions.RequestException):
                time.sleep(2 ** attempt)
        return ''

    def generate_completion(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.0,
    ):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'max_tokens': max_new_tokens,
            'temperature': temperature
        }
        
        for attempt in range(3):
            try:
                response = self.session.post(
                    f'{self.api_endpoint}/v1/completions',
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                if response.status_code != 200:
                    time.sleep(2 ** attempt)
                    continue
                data = response.json()
                return data['choices'][0]['text']
            except (requests.exceptions.JSONDecodeError, KeyError, requests.exceptions.RequestException):
                time.sleep(2 ** attempt)
        return ''

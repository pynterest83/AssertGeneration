import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


class OracleGenerator:
    def __init__(self, model_path, tokenizer_path=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        if tokenizer_path is None:
            tokenizer_path = "codeparrot/codeparrot-small-multi"
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        eos_token = self.tokenizer.special_tokens_map.get("eos_token") or \
                    self.tokenizer.special_tokens_map.get("sep_token")
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(eos_token)
        
        sep_token = self.tokenizer.special_tokens_map.get("sep_token")
        if sep_token:
            self.sep_token_id = self.tokenizer.convert_tokens_to_ids(sep_token)
        else:
            self.sep_token_id = self.eos_token_id
    
    def generate(self, test_prefix, focal_method=None, docstring=None):
        input_ids = self._build_input_ids(test_prefix, focal_method, docstring)
        input_tensor = torch.LongTensor([input_ids]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_new_tokens=200,
                num_beams=5,
                num_return_sequences=1,
                early_stopping=False,
                pad_token_id=self.eos_token_id,
                eos_token_id=self.eos_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][input_tensor.size(1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated.strip()
    
    def _build_input_ids(self, test_prefix, focal_method, docstring):
        # Tokenize test_prefix
        raw_test_prefix = self.tokenizer.decode(
            self.tokenizer.encode(test_prefix), 
            skip_special_tokens=True
        )
        test_prefix_ids = self.tokenizer.encode(raw_test_prefix, add_special_tokens=False)
        
        # Start with test_prefix + sep
        input_ids = test_prefix_ids + [self.sep_token_id]
        
        # Add docstring if provided (tc_mut_doc format: tc + sep + doc + sep + mut)
        if docstring:
            docstring_ids = self.tokenizer.encode(docstring, add_special_tokens=False)
            input_ids = input_ids + docstring_ids + [self.sep_token_id]
        
        # Add focal_method if provided
        if focal_method:
            raw_focal = self.tokenizer.decode(
                self.tokenizer.encode(focal_method), 
                skip_special_tokens=True
            )
            focal_ids = self.tokenizer.encode(raw_focal, add_special_tokens=False)
            input_ids = input_ids + focal_ids
        
        return input_ids


def load_oracle_generator(model_path=None):
    if model_path is None:
        model_path = Path(__file__).parent.parent / \
                     'RQ2/inference/run_inference/codeparrot_tc_mut_doc'
    
    return OracleGenerator(str(model_path))


def generate_oracle(test_prefix, focal_method=None, docstring=None, generator=None):
    if generator is None:
        generator = load_oracle_generator()
    
    return generator.generate(test_prefix, focal_method, docstring)
import pickle
import time
import torch
import os
from dataset_batch import Dataset, custom_collator, BatchDecodingDataset, batch_decoding_collator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers import logging as hf_logging
from accelerate import Accelerator
import numpy as np
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
import random

# Create the custom configuration
process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours

# Instantiate Accelerator with the custom configuration
accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

#accelerator = Accelerator()
hf_logging.set_verbosity_error()
torch.manual_seed(7)
test_batch_size = 5


model_data = [
    {'backbone': 'microsoft/CodeGPT-small-java', 'model_name': 'codegpt'}, #60, 50
    {'backbone': 'codeparrot/codeparrot-small-multi', 'model_name': 'codeparrot'}, #60, 60
    {'backbone': 'NinedayWang/PolyCoder-0.4B', 'model_name': 'polycoder'}, #30, 20
    {'backbone': 'Salesforce/codegen-350M-multi', 'model_name': 'codegen-350M'}, #30, 20
    {'backbone': 'microsoft/phi-1', 'model_name': 'phi-1'}, #8, 8
    {'backbone': 'Salesforce/codegen-2B-multi', 'model_name': 'codegen-2B'},
    {'backbone': 'NinedayWang/PolyCoder-2.7B', 'model_name': 'polycoder-2B'}
]

backbone = 'NinedayWang/PolyCoder-2.7B'
model_name = 'polycoder-2B'

loaded_data_names = [
   #'tc_only'
   'tc_doc'  #
   #'tc_mutsig' #
   #'tc_mut'
   #'tc_mut_doc'
   #'tc_mutsig_doc' #
]
for loaded_data_name in loaded_data_names:
    
    print("running:", loaded_data_name, backbone, model_name)
    
    if model_name == 'phi-1':
        tokenizer = AutoTokenizer.from_pretrained(backbone,  trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(backbone)
    eos_token = tokenizer.special_tokens_map.get("eos_token") or tokenizer.special_tokens_map.get("sep_token")
    print("EOS token:", eos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    print("EOS token ID:", eos_token_id)
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    sep_token = tokenizer.special_tokens_map.get("sep_token")
    if sep_token is None:
        sep_token = eos_token
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)

    if sep_id is None:
        if 'NinedayWang/PolyCoder' in backbone:
            sep_id = 2
        else:
            sep_id = eos_id

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'fine-tuning', 'dataset', f'dataset_{model_name}.pickle')
    loaded_data = pickle.load(open(dataset_path, 'rb'))
    test, valid, train = loaded_data[loaded_data_name]
    valid_dataset = BatchDecodingDataset(valid, max_len=512)
    print(len(valid_dataset))

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=batch_decoding_collator)

    def eval_model(data_loader, model, N):
        model.eval()
        exact_match =  torch.tensor([1]).cuda()
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                labels = data['labels'].cuda()
                input_ids = data['input_ids'].cuda()
                attention_mask = data['attention_mask'].cuda()

                outputs = accelerator.unwrap_model(model).generate(
                    input_ids, attention_mask=attention_mask, max_new_tokens=384, num_beams=5, num_return_sequences=1,
                    early_stopping=False, pad_token_id=eos_token_id, eos_token_id=eos_token_id
                )

                for j in range(input_ids.size(0)):
                    output = tokenizer.decode(outputs[j][input_ids.size(1): ], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    target = tokenizer.decode(labels[j][labels[j] != -100], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                
                    if output.strip() == target.strip():
                        exact_match[0] += 1
                        #print("output==============",output)
                        #print("target==============",target)


        total_match = accelerator.gather_for_metrics(exact_match)
        print(total_match)
        return  np.sum(total_match.cpu().numpy()) / N


    if model_name == 'phi-1':
        model = AutoModelForCausalLM.from_pretrained(model_name+'_'+loaded_data_name,  trust_remote_code=True).cuda()

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name+'_'+loaded_data_name).cuda()

    #optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5)
    #model, valid_loader, optimizer = accelerator.prepare(model, valid_loader, optimizer)

    model, valid_loader = accelerator.prepare(model, valid_loader)
    final_valid_accuracy = eval_model(valid_loader, model, len(valid_dataset))
    print('final_valid_accuracy: {}'.format(final_valid_accuracy))

    with open(model_name+'_'+loaded_data_name+'.txt', 'a') as wp:
            wp.write('final valid accuracy: {}'.format(final_valid_accuracy) + '\n')
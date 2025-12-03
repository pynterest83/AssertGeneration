import pickle
import time
import torch
from dataset_batch import Dataset, custom_collator, BatchDecodingDataset, batch_decoding_collator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, AutoConfig
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

train_batch = 10 
test_batch = 10
epochs = 15
max_length = 512


model_data = [
    {'backbone': 'microsoft/CodeGPT-small-java', 'model_name': 'codegpt'}, #60, 50
    {'backbone': 'codeparrot/codeparrot-small-multi', 'model_name': 'codeparrot'}, #60, 60
    {'backbone': 'NinedayWang/PolyCoder-0.4B', 'model_name': 'polycoder'}, #30, 20
    {'backbone': 'Salesforce/codegen-350M-multi', 'model_name': 'codegen-350M'}, #30, 20
    {'backbone': 'microsoft/phi-1', 'model_name': 'phi-1'}, #10, 5, 2048
    {'backbone': 'Salesforce/codegen-2B-multi', 'model_name': 'codegen-2B'}, #10, 10, DS
    {'backbone': 'NinedayWang/PolyCoder-2.7B', 'model_name': 'polycoder-2B'}#10, 10, DS
]

backbone = 'NinedayWang/PolyCoder-2.7B'
model_name = 'polycoder-2B'

loaded_data_names = [
   #'tc_only'  
   #'tc_doc' 
   #'tc_mutsig' 
   #'tc_mut' 
   #'tc_mut_doc' 
   'tc_mutsig_doc' 
]

for loaded_data_name in loaded_data_names:
    
    print("running:", loaded_data_name, backbone, model_name)

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    eos_token = tokenizer.special_tokens_map.get("eos_token") or tokenizer.special_tokens_map.get("sep_token")
    print("EOS token:", eos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    print("EOS token ID:", eos_token_id)
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    sep_token = tokenizer.special_tokens_map.get("sep_token")
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)

    if sep_id is None:
        if 'NinedayWang/PolyCoder' in backbone:
            sep_id = 2
        else:
            sep_id = eos_id

    model = AutoModelForCausalLM.from_pretrained(backbone).cuda()
    loaded_data = pickle.load(open('dataset_'+model_name+'.pickle', 'rb'))
    test, valid, train = loaded_data[loaded_data_name]

    test_dataset = BatchDecodingDataset(test, max_len=max_length)
    valid_dataset = BatchDecodingDataset(valid, max_len=max_length)
    train_dataset = Dataset(train, max_len=max_length)

    #train_dataset = train_dataset[0:1000]
    #valid_dataset = valid_dataset[0:100]
    #test_dataset = test_dataset[0:100]


    print(len(test_dataset), len(valid_dataset), len(train_dataset))


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch, shuffle=False, num_workers=0, pin_memory=True, 
        sampler=torch.utils.data.SequentialSampler(train_dataset), collate_fn=custom_collator
    )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=False, collate_fn=batch_decoding_collator)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=test_batch, shuffle=False, collate_fn=batch_decoding_collator)


    def eval_model(data_loader, model, N):
        model.eval()
        exact_match =  torch.tensor([1]).cuda()
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                labels = data['labels'].cuda()
                input_ids = data['input_ids'].cuda()
                attention_mask = data['attention_mask'].cuda()

                outputs = accelerator.unwrap_model(model).generate(
                    input_ids, attention_mask=attention_mask, max_new_tokens=300, num_beams=5, num_return_sequences=1,
                    early_stopping=False, pad_token_id=eos_token_id, eos_token_id=eos_token_id
                )

                for j in range(input_ids.size(0)):
                    output = tokenizer.decode(outputs[j][input_ids.size(1): ], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    target = tokenizer.decode(labels[j][labels[j] != -100], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    if output.strip() == target.strip():
                        exact_match[0] += 1
                        print("output==============",output)
                        print("target==============",target)


        total_match = accelerator.gather_for_metrics(exact_match)
        print(total_match)
        return  np.sum(total_match.cpu().numpy()) / N


    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=1e3, num_training_steps=int(epochs * len(train_loader))
    )

    model,test_loader, valid_loader, optimizer, scheduler = accelerator.prepare(
                model, test_loader, valid_loader, optimizer, scheduler)

    max_valid_accuracy = 0  
    c=0
    for epoch in range(epochs):
        model.train()
        train_loss = []
        start_time = time.time()
        for i, data in enumerate(train_loader):
            data = {
                'input_ids': data['input_ids'].cuda(),
                'labels': data['labels'].cuda(),
                'attention_mask': data['attention_mask'].cuda()
            }
            
            optimizer.zero_grad()
            loss = model(input_ids=data['input_ids'], labels=data['labels'], 
                        attention_mask=data['attention_mask'], return_dict=True).loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())

            if i % 200 == 0:
                # print training states every 2000 steps
                torch.cuda.empty_cache()
                print('epoch: {}, step: {}/{}, loss: {}, lr: {}, time: {}s'.format(
                    epoch + 1, i, len(train_loader), 
                    round(sum(train_loss) / len(train_loss), 4), 
                    round(scheduler.get_last_lr()[0], 8),
                    int(time.time() - start_time)
                ))
                start_time = time.time()
                train_loss = []
                
                if c ==1:
                    test_accuracy = eval_model(test_loader, model, len(test_dataset))
                    print('test accuracy: {}'.format(test_accuracy))
                    with open(model_name+'_'+loaded_data_name+'.txt', 'a') as wp:
                        wp.write('test accuracy: {}'.format(test_accuracy) + '\n')
                    c=0
                
            
        test_accuracy = eval_model(test_loader, model, len(test_dataset))
        print('test accuracy: {}'.format(test_accuracy))
        with open(model_name+'_'+loaded_data_name+'.txt', 'a') as wp:
            wp.write('test accuracy: {}'.format(test_accuracy) + '\n')

        # only save the model when validation accuracy increases
        if test_accuracy > max_valid_accuracy:
            max_valid_accuracy = test_accuracy
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                model_name+'_'+loaded_data_name,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model))
            print('checkpoint saved')

    
    
    #exit()
    #eval_accelerator = Accelerator()
    #test_batch_size = test_batch
    #valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=batch_decoding_collator)
    #model = AutoModelForCausalLM.from_pretrained(model_name+'_'+loaded_data_name).cuda()
    #model, valid_loader, optimizer = eval_accelerator.prepare(model, valid_loader, optimizer)
    
    
    #final_valid_accuracy = eval_model(valid_loader, model, len(valid_dataset))
    #print('final_valid_accuracy: {}'.format(final_valid_accuracy))

    #with open(model_name+'_'+loaded_data_name+'.txt', 'a') as wp:
            #wp.write('final valid accuracy: {}'.format(final_valid_accuracy) + '\n')
import pickle, torch, os, json
from dataset_bf import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
import utils as util
import pandas as pd
import re

# Load the CSV file
fail_catch_re = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE|re.DOTALL)

def eval_model(valid_dataset, model, dir, model_name, loaded_data_name):
    results = []
    model.eval()
    total_exact_match = set()

    for i, data in enumerate(valid_dataset):
        with torch.no_grad():

            label = data['labels'].squeeze(0).cuda()
            input_id = data['input_ids'].cuda()  #input + output 
            prompt_id = input_id[:, label == -100] #input only 
            id = data['id']

            outputs = model.generate(
                prompt_id, max_new_tokens=100, num_beams=5, num_return_sequences=1,
                early_stopping=False, pad_token_id=eos_token_id, eos_token_id=eos_token_id, return_dict_in_generate=True, output_scores=True
            )
            
            target = tokenizer.decode(label[label != -100], skip_special_tokens=True, clean_up_tokenization_spaces=True)  #target output
            output = tokenizer.decode(outputs[0][0][prompt_id.size(1): ], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            id_j = id
            score = outputs.sequences_scores
            
            exact_match=False
            
            if output.strip() == target.strip():
                total_exact_match.add(id)
                exact_match=True
            
            results.append({
                'id':id_j,
                'target':target,
                'output':output,
                'score': score.item(),
                'exact_match':exact_match})
            
            with open(dir+'/'+model_name+'_'+loaded_data_name+'.json', 'w') as f:
                        json.dump(results, f, indent=4)
    return len(total_exact_match), results


hf_logging.set_verbosity_error()
torch.manual_seed(7)

model_data = [
    {'backbone': 'codeparrot/codeparrot-small-multi', 'model_name': 'codeparrot'}
]

backbone = 'codeparrot/codeparrot-small-multi'
model_name = 'codeparrot'

loaded_data_names = [
   'tc_mut'
]


input_dir = '../input_data'
model_dir = '../models'
results_dir = '../results'


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
        if backbone == 'NinedayWang/PolyCoder-0.4B':
            sep_id = 2
        else:
            sep_id = eos_id



    loaded_data = pickle.load(open(input_dir+'/defects4j_'+model_name+'.pickle', 'rb'))
    valid = loaded_data[loaded_data_name][0]
    print(len(valid['inputs']))

    valid_dataset = Dataset(valid, max_len=600)
    print(len(valid_dataset))
    model = AutoModelForCausalLM.from_pretrained(model_dir+'/'+model_name+'_'+loaded_data_name).cuda()


    if not os.path.exists(results_dir+'/'+model_name+'_'+loaded_data_name):
        os.makedirs(results_dir+'/'+model_name+'_'+loaded_data_name)
    if not os.path.exists(results_dir+'/'+model_name+'_'+loaded_data_name):  
        print("path still not exists")

    exact_match, results = eval_model(valid_dataset, model, results_dir+'/'+model_name+'_'+loaded_data_name, model_name, loaded_data_name)
    print("exact match:", exact_match)

    # Write the results to a .json file
    with open(results_dir+'/'+model_name+'_'+loaded_data_name+'/'+model_name+'_'+loaded_data_name+'.json', 'w') as f:
            json.dump(results, f, indent=4)

    #construct the oracle_preds.csv

    # now copy a oracle_pred in this directory and start modifying
    util.copy_csv_file(input_dir+'/oracle_preds.csv', results_dir+'/'+model_name+'_'+loaded_data_name+"/oracle_preds.csv")


    # Read the CSV file into a DataFrame
    df = pd.read_csv(results_dir+'/'+model_name+'_'+loaded_data_name+"/oracle_preds.csv")

    data = util.read_json(results_dir+'/'+model_name+'_'+loaded_data_name+'/'+model_name+'_'+loaded_data_name+'.json')  #read the actual prediction file

    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
       
        id = row['id'] 
    
        if util.get_values_by_id(data, id) is None:
            df.drop(index, inplace=True)

        else:

            target, output, score, exact_match = util.get_values_by_id(data, id)

            if output == "exception":   #converting to toga format
                df.at[index, 'except_pred'] = 1
                df.at[index, 'assert_pred'] = ""
            else:
                df.at[index, 'except_pred'] = 0
                df.at[index, 'assert_pred'] = str(output)

                if not str(output).endswith(";"):
                     df.at[index, 'assert_pred'] = ""
                     
           
    df.to_csv(results_dir+'/'+model_name+'_'+loaded_data_name+"/oracle_preds.csv", index=False)






 
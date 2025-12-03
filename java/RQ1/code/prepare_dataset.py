from transformers import AutoTokenizer
import pickle
import pandas as pd
import re


def clean_data(data):
    lines = data.strip().split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    cleaned_lines = [lines[0].strip()] + ['\t' + line.lstrip() for line in non_empty_lines[1:-1]] + [non_empty_lines[-1].strip()]
    cleaned_data = '\n'.join(cleaned_lines)
    return cleaned_data

def remove_empty_line(data):
    lines = data.strip().split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    cleaned_data = '\n'.join(non_empty_lines)
    return cleaned_data


count_empty_output=0
model_data = [
    {'backbone': 'microsoft/CodeGPT-small-java', 'model_name': 'codegpt'}, # 60, 50
    {'backbone': 'codeparrot/codeparrot-small-multi', 'model_name': 'codeparrot'}, # 50, 40
    {'backbone': 'Salesforce/codegen-350M-multi', 'model_name': 'codegen-350M'}, # 30, 20
    {'backbone': 'NinedayWang/PolyCoder-0.4B', 'model_name': 'polycoder'}, # 30, 20
    {'backbone': 'microsoft/phi-1', 'model_name': 'phi-1'}, # 8, 8
    {'backbone': 'Salesforce/codegen-2B-multi', 'model_name': 'codegen-2B'},
    {'backbone': 'NinedayWang/PolyCoder-2.7B', 'model_name': 'polycoder-2B'}
]

for data in model_data:
    backbone = data['backbone']
    model_name = data['model_name']
    print("running ",backbone)

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

    inputs = pd.read_csv('inputs.csv') 
    inputs = inputs.sample(frac = 1)
    meta = pd.read_csv('meta.csv')
    N_input = len(inputs)
    N_meta = len(meta)
    print(N_input, N_meta)
    N=N_input
    

    #we can use different type of information 
    tc_only = {'inputs': [], 'outputs': []}
    tc_mut = {'inputs': [], 'outputs': []}
    tc_mutsig = {'inputs': [], 'outputs': []}
    tc_doc = {'inputs': [], 'outputs': []}
    tc_mut_doc = {'inputs': [], 'outputs': []}  
    tc_mutsig_doc = {'inputs': [], 'outputs': []}


    # Iterate through both DataFrames simultaneously
    for (index_inputs, row_inputs), (index_meta, row_meta) in zip(inputs.iterrows(), meta.iterrows()):
        # Assuming both DataFrames have the same number of rows
        # You can access values in each row using column names
        mut = row_inputs['focal_method']

        if "nan" in str(mut):
            mut=""
            signature = ""

        else:
            # Split the string by line breaks
            lines = str(mut).split('\n')
            # Get the first line
            signature = lines[0]


        test_prefix = row_inputs['test_prefix']

        #do some processing 
        # Define a regular expression pattern to match lines with "Undeclared exception!" comments
        pattern = r'^.*\/\/\s*Undeclared exception!.*$'
        # Split the input text into lines
        lines = test_prefix.split('\n')
        # Filter out lines with "Undeclared exception!" comments and empty lines
        filtered_lines = [line for line in lines if not re.match(pattern, line) and line.strip()]
        # Join the filtered lines back together
        test_prefix = '\n'.join(filtered_lines)

        docstring = row_inputs['docstring']

        if "nan" in str(docstring):
            docstring=""

        clean_test_prefix = test_prefix
        output=""

        if "try" in str(test_prefix) and "catch" in str(test_prefix):
            clean_test_prefix = re.sub(r'try\s*\{', '', test_prefix)  # Remove try {
            clean_test_prefix = re.sub(r'fail\s*\([^)]*\);', '', clean_test_prefix)  # Remove fail();
            clean_test_prefix = re.sub(r'\}\s*catch\s*\([^)]*\)\s*\{.*?\}', '', clean_test_prefix, flags=re.DOTALL)  # Remove catch {...}
            clean_test_prefix = clean_data(clean_test_prefix)
            output = "exception"

        else:
            if "assert" in str(test_prefix):
                assert_re = re.compile("assert\w*\(.*\);")
                assertion_lines = re.findall(assert_re, test_prefix)
                clean_test_prefix = re.sub(assert_re, '', test_prefix)
                clean_test_prefix = remove_empty_line(clean_test_prefix)
                if len(assertion_lines)>0:
                    print("assert======================",assertion_lines[0])
                    output = str(assertion_lines[0])
        
        
        if len(output.strip())==0:
            continue
            count_empty_output+=1
            print(".......................start................")
            print("---------------------------------------------")
            print("clean_test_prefix==========",clean_test_prefix)
            print("---------------------------------------------")
            print("focal_method==========",focal_method)
            print("---------------------------------------------")
            print("signature==========",signature)
            print("---------------------------------------------")
            print("docstring==========",docstring)
            print("---------------------------------------------")
            print("output==========",output)
            print(".......................end................")

        else:
            print("test_prefix==========",clean_test_prefix)
            print("output==========",output)
           

        raw_test_prefix = tokenizer.decode(tokenizer.encode(clean_test_prefix), skip_special_tokens=True)
        raw_mut = tokenizer.decode(tokenizer.encode(mut), skip_special_tokens=True)    
      
        test_prefix = tokenizer.encode(raw_test_prefix)
        mut = tokenizer.encode(raw_mut)
        docstring = tokenizer.encode(docstring)
        output = tokenizer.encode(output)
        signature = tokenizer.encode(signature)

        # input: tc_only
        x = test_prefix + [sep_id]
        # output: fix
        y = output + [eos_id]
        
        tc_only['inputs'].append(x)
        tc_only['outputs'].append(y)

        # input: tc_doc
        x = test_prefix + [sep_id] + docstring
        # output: fix
        y = output + [eos_id]
        
        tc_doc['inputs'].append(x)
        tc_doc['outputs'].append(y)


        # input: tc_mutsig
        x = test_prefix + [sep_id] + signature
        # output: fix
        y = output + [eos_id]
        
        tc_mutsig['inputs'].append(x)
        tc_mutsig['outputs'].append(y)

        # input: tc_mut
        x = test_prefix + [sep_id] + mut
        # output: fix
        y = output + [eos_id]
        
        tc_mut['inputs'].append(x)
        tc_mut['outputs'].append(y)

        # input: tc_mut_doc
        x = test_prefix + [sep_id] + docstring + [sep_id] + mut
        # output: fix
        y = output + [eos_id]
        
        tc_mut_doc['inputs'].append(x)
        tc_mut_doc['outputs'].append(y)

        # input: tc_mutsig_doc
        x = test_prefix + [sep_id] + docstring + [sep_id] + signature 
        # output: fix
        y = output + [eos_id]
        
        tc_mutsig_doc['inputs'].append(x)
        tc_mutsig_doc['outputs'].append(y)


    test_size = int(N * 0.05)
    valid_size = int(N * 0.05)
    valid_end_index = test_size + valid_size

    tc_only_test = {key: tc_only[key][: test_size] for key in ('inputs', 'outputs')}
    tc_only_valid = {key: tc_only[key][test_size: valid_end_index] for key in ('inputs', 'outputs')}
    tc_only_train = {key: tc_only[key][valid_end_index: ] for key in ('inputs', 'outputs')}

    tc_doc_test = {key: tc_doc[key][: test_size] for key in ('inputs', 'outputs')}
    tc_doc_valid = {key: tc_doc[key][test_size: valid_end_index] for key in ('inputs', 'outputs')}
    tc_doc_train = {key: tc_doc[key][valid_end_index: ] for key in ('inputs', 'outputs')}

    tc_mutsig_test = {key: tc_mutsig[key][: test_size] for key in ('inputs', 'outputs')}
    tc_mutsig_valid = {key: tc_mutsig[key][test_size: valid_end_index] for key in ('inputs', 'outputs')}
    tc_mutsig_train = {key: tc_mutsig[key][valid_end_index: ] for key in ('inputs', 'outputs')}

    tc_mut_test = {key: tc_mut[key][: test_size] for key in ('inputs', 'outputs')}
    tc_mut_valid = {key: tc_mut[key][test_size: valid_end_index] for key in ('inputs', 'outputs')}
    tc_mut_train = {key: tc_mut[key][valid_end_index: ] for key in ('inputs', 'outputs')}

    tc_mut_doc_test = {key: tc_mut_doc[key][: test_size] for key in ('inputs', 'outputs')}
    tc_mut_doc_valid = {key: tc_mut_doc[key][test_size: valid_end_index] for key in ('inputs', 'outputs')}
    tc_mut_doc_train = {key: tc_mut_doc[key][valid_end_index: ] for key in ('inputs', 'outputs')}

    tc_mutsig_doc_test = {key: tc_mutsig_doc[key][: test_size] for key in ('inputs', 'outputs')}
    tc_mutsig_doc_valid = {key: tc_mutsig_doc[key][test_size: valid_end_index] for key in ('inputs', 'outputs')}
    tc_mutsig_doc_train = {key: tc_mutsig_doc[key][valid_end_index: ] for key in ('inputs', 'outputs')}

   
    with open('dataset_'+model_name+'.pickle', 'wb') as f:
        pickle.dump({
            'tc_only': [tc_only_test, tc_only_valid, tc_only_train],
            'tc_doc': [tc_doc_test, tc_doc_valid, tc_doc_train],
            'tc_mutsig': [tc_mutsig_test, tc_mutsig_valid, tc_mutsig_train], #new addition
            'tc_mut': [tc_mut_test, tc_mut_valid, tc_mut_train],
            'tc_mut_doc': [tc_mut_doc_test, tc_mut_doc_valid, tc_mut_doc_train],
            'tc_mutsig_doc': [tc_mutsig_doc_test, tc_mutsig_doc_valid, tc_mutsig_doc_train]
        }, f)


    print(count_empty_output)
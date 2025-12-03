from transformers import AutoTokenizer
import pickle, csv
import pandas as pd
import re, os

fail_catch_re = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE|re.DOTALL)
    

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

model_data = [
    {'backbone': 'codeparrot/codeparrot-small-multi', 'model_name': 'codeparrot'}
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

    evo_directory = '../evosuite_reaching_tests'
    out_directory = '../input_data'


    inputs = pd.read_csv(evo_directory+'/inputs.csv') 
    meta = pd.read_csv(evo_directory+'/meta.csv') 
    N_input = len(inputs)
    N_meta = len(meta)
    assert N_input == N_meta
    print(N_input, N_meta)
    N=N_input


    tc_mut = {'inputs': [], 'outputs': [], 'id': [], 'test_prefix': []}
    tc_mut_doc = {'inputs': [], 'outputs': [], 'id': [], 'test_prefix': []}


    excep_oracle_rows = []
    assert_oracle_rows = []
 
    id = 1
    
    for (index_inputs, row_inputs), (index_meta, row_meta) in zip(inputs.iterrows(), meta.iterrows()):

        exception_prefix = re.search(fail_catch_re, row_inputs['test_prefix'])

       
    
        mut = row_inputs['focal_method']

        if "nan" in str(mut): 
            mut=""

        GT_output = row_meta['exception_lbl']
        test_prefix = row_inputs['test_prefix']

    

        if GT_output is False:

            output = row_meta['assertion_lbl']

            if "nan" in str(output): 

                output=""
        else:

            output = "exception"

        GT_output = output

        docstring = row_inputs['docstring']

        if "nan" in str(docstring):
            docstring=""
        
        original_test_prefix = test_prefix

        pattern = r'^.*\/\/\s*Undeclared exception!.*$'
        lines = test_prefix.split('\n')
        filtered_lines = [line for line in lines if not re.match(pattern, line) and line.strip()]
        test_prefix = '\n'.join(filtered_lines)
    
        clean_test_prefix = test_prefix
    
        if "try" in str(test_prefix) and "catch" in str(test_prefix):
            clean_test_prefix = re.sub(r'try\s*\{', '', test_prefix)  # Remove try {
            clean_test_prefix = re.sub(r'fail\s*\([^)]*\);', '', clean_test_prefix)  # Remove fail();
            clean_test_prefix = re.sub(r'\}\s*catch\s*\([^)]*\)\s*\{.*?\}', '', clean_test_prefix, flags=re.DOTALL)  # Remove catch {...}
            clean_test_prefix = clean_data(clean_test_prefix)
        
        
        else:
            if "assert" in str(test_prefix):
                assert_re = re.compile("assert\w*\(.*\);")
                assertion_lines = re.findall(assert_re, test_prefix)
                clean_test_prefix = re.sub(assert_re, '', test_prefix)
                clean_test_prefix = remove_empty_line(clean_test_prefix)

       
        if exception_prefix:
            #print("clean test case:", clean_test_prefix)
            #assert len(clean_test_prefix)>0
            excep_oracle_rows.append([id, row_meta['project'], row_meta['bug_num'], row_meta['test_name'], row_inputs['test_prefix'], -1, "nan", clean_test_prefix])            
        else:
            assert_oracle_rows.append([id, row_meta['project'], row_meta['bug_num'], row_meta['test_name'], row_inputs['test_prefix'], -1, "nan", clean_test_prefix])

        raw_test_prefix = tokenizer.decode(tokenizer.encode(clean_test_prefix), skip_special_tokens=True)
        raw_mut = tokenizer.decode(tokenizer.encode(mut), skip_special_tokens=True)    
    
        test_prefix = tokenizer.encode(raw_test_prefix)
        mut = tokenizer.encode(raw_mut)
        docstring = tokenizer.encode(docstring)
        output = tokenizer.encode(GT_output)
        

        # input: tc_mut
        x = test_prefix + [sep_id] + mut
        # output: fix
        y = output + [eos_id]
        
        tc_mut['inputs'].append(x)
        tc_mut['outputs'].append(y)
        tc_mut['id'].append(id)
        tc_mut['test_prefix'].append(test_prefix + [sep_id])

        # input: tc_mut_doc
        x = test_prefix + [sep_id] + docstring + [sep_id] + mut
        # output: fix
        y = output + [eos_id]
        
        tc_mut_doc['inputs'].append(x)
        tc_mut_doc['outputs'].append(y)
        tc_mut_doc['id'].append(id)
        tc_mut_doc['test_prefix'].append(test_prefix)

        id+=1

    
    with open(out_directory+'/oracle_preds.csv', mode='w') as f1:
        writer = csv.writer(f1)
        writer.writerow('id,project,bug_num,test_name,test_prefix,except_pred,assert_pred, clean_test_prefix'.split(','))
        writer.writerows(excep_oracle_rows)
        writer.writerows(assert_oracle_rows)


    with open(out_directory+'/defects4j_'+model_name+'.pickle', 'wb') as f:
        pickle.dump({
            'tc_mut': [tc_mut],
            'tc_mut_doc': [tc_mut_doc]
        }, f)
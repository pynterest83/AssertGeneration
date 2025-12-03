import pandas as pd
import re

# Load the CSV file
fail_catch_re = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE|re.DOTALL)

file_path = 'results/TOGLL/oracle_preds.csv'
oracle_preds_exception = pd.read_csv(file_path)
oracle_preds_assertion = pd.read_csv(file_path)
oracle_preds_prefix_only = pd.read_csv(file_path)

for index, row in oracle_preds_exception.iterrows():
    
    exception_prefix = re.search(fail_catch_re,  oracle_preds_exception.at[index, 'test_prefix'])
    
    if not exception_prefix :
        oracle_preds_exception.drop(index, inplace=True)

   
for index, row in oracle_preds_assertion.iterrows():
    
    exception_prefix = re.search(fail_catch_re,  oracle_preds_assertion.at[index, 'test_prefix'])
    
    if exception_prefix or "nan" in str(oracle_preds_prefix_only.at[index, 'assert_pred']):
        oracle_preds_assertion.drop(index, inplace=True)
        
   
for index, row in oracle_preds_prefix_only.iterrows():
    
    exception_prefix = re.search(fail_catch_re,  oracle_preds_prefix_only.at[index, 'test_prefix'])
    
    if exception_prefix:
        oracle_preds_prefix_only.drop(index, inplace=True)

    else:

        if "nan" not in str(oracle_preds_prefix_only.at[index, 'assert_pred']):
            oracle_preds_prefix_only.drop(index, inplace=True)
      

oracle_preds_assertion.to_csv('results/TOGLL/assertion_only/oracle_preds.csv', index=False)
oracle_preds_exception.to_csv('results/TOGLL/exception_only/oracle_preds.csv', index=False)
oracle_preds_prefix_only.to_csv('results/TOGLL/prefix_only/oracle_preds.csv', index=False)

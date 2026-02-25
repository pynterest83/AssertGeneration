import pandas as pd
import csv

# Source: TOGA oracle_preds.csv  
toga_df = pd.read_csv('toga-reflect/artifact/RQ2/toga-model-inputs-outputs/async-http-client/toga_output/oracle_preds.csv')
toga_prefix_dict = dict(zip(toga_df['test_name'], toga_df['test_prefix']))

print(f"TOGA: {len(toga_df)} rows")
print(f"TOGA assertions: {toga_df['test_prefix'].str.contains('assert', case=False, na=False).sum()}")

# Verify TOGA data
sample = toga_prefix_dict['org.asynchttpclient.extras.guava.ListenableFutureAdapter_ESTest::test0']
print(f"Sample has assert: {'assert' in sample.lower()}")

target_files = [
    'data/RQ1/output/async-http-client/oracle_preds_external.csv',
    'data/RQ1/output/async-http-client/oracle_preds_external_DeepSeek-Coder-V2-Lite-Instruct.csv',
    'data/RQ1/output/async-http-client/oracle_preds_external_returntype_DeepSeek-Coder-V2-Lite-Instruct.csv',
    'data/RQ1/output/async-http-client/oracle_preds_returntype_DeepSeek-Coder-V2-Lite-Instruct.csv',
]

for target_file in target_files:
    df = pd.read_csv(target_file)
    fname = target_file.split('/')[-1]
    
    # Update test_prefix using TOGA
    df['test_prefix'] = df['test_name'].apply(lambda x: toga_prefix_dict.get(x, ''))
    
    has_assert = df['test_prefix'].str.contains('assert', case=False, na=False).sum()
    print(f"\n{fname}: {has_assert}/{len(df)} with assert (before save)")
    
    # Save with explicit quoting
    df.to_csv(target_file, index=False, quoting=csv.QUOTE_ALL)
    
    # Re-read to verify
    df_check = pd.read_csv(target_file)
    has_assert_check = df_check['test_prefix'].str.contains('assert', case=False, na=False).sum()
    print(f"  After save/reload: {has_assert_check}/{len(df_check)} with assert")

print("\nDone!")

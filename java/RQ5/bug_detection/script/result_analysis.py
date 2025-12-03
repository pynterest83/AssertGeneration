import pandas as pd
import re
import os, glob

fail_catch_re = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE|re.DOTALL)

def analyze_results(path):
    oracle_preds = path+'/oracle_preds.csv'
    oracle_preds_df = pd.read_csv(oracle_preds)

    test_prefixes = oracle_preds_df['test_prefix']
    except_pred = oracle_preds_df['except_pred']
    assert_pred = oracle_preds_df['assert_pred']
    projects = oracle_preds_df['project']
    bug_nums = oracle_preds_df['bug_num']


    total_prefix=0
    total_prefix_with_assertion_oracle=0
    total_prefix_with_exception_oracle=0

    total_exception_prefix_misclassified=0
    total_exception_prefix_correctly_classified=0

    total_assertion_prefix_misclassified=0
    total_assertion_prefix_correctly_classified=0
    total_assertion_generated=0
    total_empty_assertion_generated=0


    for prefix,ex_pred,as_pred,project,bug_num in zip(test_prefixes,except_pred,assert_pred,projects,bug_nums):
        total_prefix+=1

        exception_prefix = re.search(fail_catch_re, prefix)
        if exception_prefix:
            total_prefix_with_exception_oracle+=1
            if "0" in str(ex_pred):
                total_exception_prefix_misclassified+=1
            else:
                total_exception_prefix_correctly_classified+=1
        else:
            total_prefix_with_assertion_oracle+=1

            if "1" in str(ex_pred):
                total_assertion_prefix_misclassified+=1
            else:
                total_assertion_prefix_correctly_classified+=1

                if "assert" in str(as_pred):
                    total_assertion_generated+=1
                else:
                    total_empty_assertion_generated+=1

    test_data_exception_prefix = path+'/exception_prefix/'+path.lower()+'_generated/test_data.csv'
    test_data_exception_prefix_df = pd.read_csv(test_data_exception_prefix)
    exception_bugs = test_data_exception_prefix_df[test_data_exception_prefix_df['TP'] == True][['project', 'bug_num']].drop_duplicates()
    unique_bugs_excep = set(exception_bugs.itertuples(index=False, name=None))
    print("total exception bugs found by "+ path+':', len(unique_bugs_excep))




    test_data_assertion_prefix = path+'/assertion_prefix/'+path.lower()+'_generated/test_data.csv'
    test_data_assertion_prefix_df = pd.read_csv(test_data_assertion_prefix)
    unique_pairs = test_data_assertion_prefix_df[test_data_assertion_prefix_df['TP'] == True][['project', 'bug_num']].drop_duplicates()
    unique_bugs_assert = set(unique_pairs.itertuples(index=False, name=None))
    print("total assertion bugs found by "+ path+':', len(unique_bugs_assert))

    
    test_data_prefix_df=path+'/prefix_only/'+path.lower()+'_generated/test_data.csv'
    test_data_prefix_df = pd.read_csv(test_data_prefix_df)
    unique_pairs_prefix = test_data_prefix_df[test_data_prefix_df['TP'] == True][['project', 'bug_num']].drop_duplicates()
    unique_bugs_prefix = set(unique_pairs_prefix.itertuples(index=False, name=None))
    print("total bugs found by prefix only: ", len(unique_bugs_prefix))

    print("total unique bugs found by "+ path+':', len(unique_bugs_assert | unique_bugs_excep | unique_bugs_prefix))

    return unique_bugs_assert , unique_bugs_excep , unique_bugs_prefix

if __name__=='__main__':
    toga = 'TOGA'
    togll = 'TOGLL'

    unique_bugs_assert_toga, unique_bugs_excep_toga, unique_bugs_prefix_toga = analyze_results(toga)
    unique_bugs_assert_togll, unique_bugs_excep_togll, unique_bugs_prefix_togll = analyze_results(togll)

    print("TOGA unique:", len((unique_bugs_assert_toga | unique_bugs_excep_toga) - (unique_bugs_assert_togll | unique_bugs_excep_togll)))
    print("TOGLL unique:", len((unique_bugs_assert_togll | unique_bugs_excep_togll) - (unique_bugs_assert_toga | unique_bugs_excep_toga)))

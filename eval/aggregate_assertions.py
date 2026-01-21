import pandas as pd
import os, re, shutil, argparse
from pathlib import Path

def aggregate_assertions(base_dir, output_dir):
    assert_re = re.compile(r"assert\w*\(.*\)")
    oracle_preds = pd.read_csv(base_dir + "/outputs/oracle_preds.csv")
    test_name = oracle_preds['test_name']
    test_prefix = oracle_preds['test_prefix']
    file_path = oracle_preds['file_path']
    assert_pred = oracle_preds['assert_pred']
    count = {}
    total_assertion_replaced = 0
    ne_assert_pred = oracle_preds.assert_pred.count()

    for prefix, fpath, tname, apred in zip(test_prefix, file_path, test_name, assert_pred):
        loc = output_dir + "/" + fpath
        loc = loc.replace(".java", ".txt")
        filename = Path(loc)
        
        os.makedirs(filename.parent, exist_ok=True)
        
        if loc not in count:
            count[loc] = 0
        else:
            count[loc] = count[loc] + 1

        with open(filename, 'a+') as split_tests:
            if count[loc] == 0:
                split_tests.truncate(0)

            split_tests.write(' @Test(timeout = 4000)\n')

            if "assert" in str(apred):
                new_assertion = str(apred)
                total_assertion_replaced = total_assertion_replaced + 1
            else:
                new_assertion = ""
            prefix = re.sub(assert_re, new_assertion, str(prefix))
            split_tests.write(prefix)
            split_tests.write('\n')

    print(total_assertion_replaced)
    print(ne_assert_pred)

def copy_assertions(base_dir, output_dir):
    oracle_preds = pd.read_csv(base_dir + "/outputs/oracle_preds.csv")
    locations = oracle_preds['file_path'].unique()
    
    for fpath in locations:
        java_test_file = output_dir + "/" + fpath
        aggregated_test_file = java_test_file.replace(".java", ".txt")
        
        if not os.path.exists(aggregated_test_file):
            continue
            
        with open(aggregated_test_file, 'r') as file:
            a_tests = file.read()
        
        with open(java_test_file, 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            f.truncate()
            for line in lines:
                if "@Test(timeout = 4000)" in line:
                    f.write(a_tests)
                    break
                f.write(line)
            f.write("}\n")
        
        os.remove(aggregated_test_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--base_dir', required=True)
    parser.add_argument('-o', '--output_dir', default=None)
    args = parser.parse_args()
    
    if args.base_dir.endswith('/'):
        args.base_dir = args.base_dir[:-1]
    
    output_dir = args.output_dir
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(args.base_dir, output_dir)
    
    aggregate_assertions(args.base_dir, output_dir)
    copy_assertions(args.base_dir, output_dir)

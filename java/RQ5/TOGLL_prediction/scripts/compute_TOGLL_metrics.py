import re
import pandas as pd

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report


fail_catch_re = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE|re.DOTALL)

oracle_preds = 'results/TOGLL/oracle_preds.csv'
oracle_preds_df = pd.read_csv(oracle_preds)

test_prefixes = oracle_preds_df['test_prefix']
except_pred = oracle_preds_df['except_pred']
assert_pred = oracle_preds_df['assert_pred']

y_true = []
y_pred = []


for prefix, ex_pred, as_pred in zip(test_prefixes,except_pred,assert_pred):
    

    exception_prefix = re.search(fail_catch_re, prefix)

    if exception_prefix:
        y_true.append(1)
    else:
        y_true.append(0)


    
    if "0" in str(ex_pred):
        y_pred.append(0)
    else:
        y_pred.append(1)

# Compute overall precision and recall
overall_precision = precision_score(y_true, y_pred, average='macro')
overall_recall = recall_score(y_true, y_pred, average='macro')

print(f'Overall Precision: {overall_precision:.2f}')
print(f'Overall Recall: {overall_recall:.2f}')

# Compute precision and recall per class
class_precision = precision_score(y_true, y_pred, average=None)
class_recall = recall_score(y_true, y_pred, average=None)

for i, (p, r) in enumerate(zip(class_precision, class_recall)):
    print(f'Class {i} - Precision: {p:.2f}, Recall: {r:.2f}')

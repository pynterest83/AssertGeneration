import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..', '..')

SOLUTION_DIR = os.path.join(ROOT, 'data', 'RQ3', 'output', 'bug_detection')
INPUT_DIR = os.path.join(ROOT, 'data', 'RQ3', 'input')

CHANNELS = ['assertion_prefix', 'exception_prefix', 'prefix_only']

# Đọc test_data.csv (output Docker), trả về set (project, bug_num) các bug có TP=True.
def get_bugs_from_test_data(test_data_path):
    if not os.path.exists(test_data_path):
        return set()
    df = pd.read_csv(test_data_path)
    tp = df[df['TP'] == True]
    return set(zip(tp['project'], tp['bug_num']))

# Quét 3 channel, đếm bugs unique mỗi channel + union 3 channel = tổng bugs phát hiện.
def analyze_approach(name, base_dir, generated_subdir):
    print(f"\n{name}")
    all_bugs = set()
    channel_bug_counts = {}
    for channel in CHANNELS:
        test_data = os.path.join(base_dir, channel, generated_subdir, 'test_data.csv')
        bugs = get_bugs_from_test_data(test_data)
        channel_bug_counts[channel] = len(bugs)
        print(f"  {channel}: {len(bugs)} bugs")
        all_bugs |= bugs

    print(f"  total_unique: {len(all_bugs)} bugs")
    return all_bugs, channel_bug_counts

# So sánh GT_output=='exception' (meta_llm) vs assert_pred NaN (pred) -> tính precision/recall/F1 cho Agent 1.
def analyze_exception_classifier():
    total_gt_exc = 0
    total_pred_exc = 0
    true_pos = 0
    false_pos = 0

    projects = [p for p in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, p))]
    for project in sorted(projects):
        meta_path = os.path.join(INPUT_DIR, project, 'infer_input', 'meta_llm.csv')
        pred_path = os.path.join(ROOT, 'data', 'RQ3', 'output', project, 'oracle_preds_Qwen3-Coder-Next.csv')
        if not os.path.exists(meta_path) or not os.path.exists(pred_path):
            continue

        meta = pd.read_csv(meta_path)
        pred = pd.read_csv(pred_path)

        gt_exc_names = set(meta[meta['GT_output'] == 'exception']['test_name'])
        pred_exc = pred[pred['assert_pred'].isna()]
        pred_exc_names = set(pred_exc['test_name'])

        tp = len(gt_exc_names & pred_exc_names)
        fp = len(pred_exc_names - gt_exc_names)
        fn = len(gt_exc_names - pred_exc_names)

        total_gt_exc += len(gt_exc_names)
        total_pred_exc += len(pred_exc_names)
        true_pos += tp
        false_pos += fp

    recall = true_pos / total_gt_exc * 100 if total_gt_exc else 0
    precision = true_pos / total_pred_exc * 100 if total_pred_exc else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\nexception_classifier")
    print(
        f"  gt={total_gt_exc}, pred={total_pred_exc}, tp={true_pos}, fp={false_pos}, "
        f"recall={recall:.1f}%, precision={precision:.1f}%, f1={f1:.1f}%"
    )

# Entry point: in tổng bugs SOLUTION detect + metrics exception_classifier.
def main():
    solution_bugs, _ = analyze_approach(
        "SOLUTION",
        SOLUTION_DIR,
        'toga_generated'
    )

    # Exception classifier
    analyze_exception_classifier()

    print(f"\nsummary: total_unique={len(solution_bugs)}")

if __name__ == '__main__':
    main()

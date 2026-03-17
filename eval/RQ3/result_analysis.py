import pandas as pd
import os
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..', '..')

SOLUTION3_DIR = os.path.join(ROOT, 'data', 'RQ3', 'output', 'bug_detection')
TOGLL_DIR = os.path.join(ROOT, 'togll', 'RQ5', 'bug_detection', 'TOGLL')
TOGA_DIR = os.path.join(ROOT, 'togll', 'RQ5', 'bug_detection', 'TOGA')
INPUT_DIR = os.path.join(ROOT, 'data', 'RQ3', 'input')

CHANNELS = ['assertion_prefix', 'exception_prefix', 'prefix_only']
PROJECTS = [
    'Cli', 'Codec', 'Compress', 'Csv', 'Gson',
    'JacksonCore', 'JacksonDatabind', 'Jsoup', 'JxPath', 'Lang', 'Math',
]


def get_bugs_from_test_data(test_data_path):
    if not os.path.exists(test_data_path):
        return set()
    df = pd.read_csv(test_data_path)
    tp = df[df['TP'] == True]
    return set(zip(tp['project'], tp['bug_num']))


def analyze_approach(name, base_dir, generated_subdir):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    all_bugs = set()
    for channel in CHANNELS:
        test_data = os.path.join(base_dir, channel, generated_subdir, 'test_data.csv')
        bugs = get_bugs_from_test_data(test_data)

        oracle_preds = os.path.join(base_dir, channel, 'oracle_preds.csv')
        n_rows = 0
        if os.path.exists(oracle_preds):
            n_rows = len(pd.read_csv(oracle_preds))

        print(f"  {channel:20s}: {n_rows:4d} tests → {len(bugs):3d} unique bugs")
        all_bugs |= bugs

    print(f"  {'TOTAL':20s}:            {len(all_bugs):3d} unique bugs")

    # Per-project breakdown
    proj_bugs = Counter()
    for p, b in all_bugs:
        proj_bugs[p] += 1
    print(f"\n  Per-project:")
    for p in PROJECTS:
        print(f"    {p:20s}: {proj_bugs.get(p, 0):3d}")

    return all_bugs


def analyze_exception_classifier():
    print(f"\n{'='*60}")
    print(f"  Exception Classifier Performance")
    print(f"{'='*60}")

    total_gt_exc = 0
    total_pred_exc = 0
    true_pos = 0
    false_pos = 0

    for project in PROJECTS:
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

    print(f"  GT exceptions:   {total_gt_exc}")
    print(f"  Pred exceptions: {total_pred_exc}")
    print(f"  True positives:  {true_pos}")
    print(f"  False positives: {false_pos}")
    print(f"  Recall:    {recall:.1f}%")
    print(f"  Precision: {precision:.1f}%")
    print(f"  F1:        {f1:.1f}%")


def main():
    print("=" * 60)
    print("  RQ3: Bug Detection Results Comparison")
    print("=" * 60)

    sol3_bugs = analyze_approach(
        "Solution_3 (Qwen3-Coder-Next)",
        SOLUTION3_DIR,
        'toga_generated'
    )

    togll_bugs = analyze_approach(
        "TOGLL",
        TOGLL_DIR,
        'togll_generated'
    )

    toga_bugs = analyze_approach(
        "TOGA",
        TOGA_DIR,
        'toga_generated'
    )

    # Comparison
    print(f"\n{'='*60}")
    print(f"  Summary Comparison")
    print(f"{'='*60}")
    print(f"  {'Approach':35s} {'Bugs':>5s}  {'vs TOGLL':>10s}  {'vs TOGA':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'Solution_3 (Qwen3-Coder-Next)':35s} {len(sol3_bugs):5d}  {len(sol3_bugs)-len(togll_bugs):+10d}  {len(sol3_bugs)-len(toga_bugs):+10d}")
    print(f"  {'TOGLL':35s} {len(togll_bugs):5d}  {'—':>10s}  {len(togll_bugs)-len(toga_bugs):+10d}")
    print(f"  {'TOGA':35s} {len(toga_bugs):5d}  {len(toga_bugs)-len(togll_bugs):+10d}  {'—':>10s}")

    # Overlap analysis
    print(f"\n  Overlap Analysis (Solution_3 vs TOGLL):")
    common = sol3_bugs & togll_bugs
    sol3_only = sol3_bugs - togll_bugs
    togll_only = togll_bugs - sol3_bugs
    print(f"    Common bugs:         {len(common)}")
    print(f"    Solution_3 only:     {len(sol3_only)}")
    print(f"    TOGLL only:          {len(togll_only)}")

    if sol3_only:
        print(f"\n    Solution_3 unique bugs:")
        for p, b in sorted(sol3_only):
            print(f"      {p}-{b}")

    if togll_only:
        print(f"\n    TOGLL unique bugs:")
        for p, b in sorted(togll_only):
            print(f"      {p}-{b}")

    # Overlap analysis (Solution_3 vs TOGA)
    print(f"\n  Overlap Analysis (Solution_3 vs TOGA):")
    common_toga = sol3_bugs & toga_bugs
    sol3_only_toga = sol3_bugs - toga_bugs
    toga_only = toga_bugs - sol3_bugs
    print(f"    Common bugs:         {len(common_toga)}")
    print(f"    Solution_3 only:     {len(sol3_only_toga)}")
    print(f"    TOGA only:           {len(toga_only)}")

    # Exception classifier
    analyze_exception_classifier()

    # Per-project comparison table
    print(f"\n{'='*60}")
    print(f"  Per-Project Comparison")
    print(f"{'='*60}")
    print(f"  {'Project':20s} {'Sol3':>5s} {'TOGLL':>6s} {'TOGA':>5s} {'Diff':>6s}")
    print(f"  {'-'*45}")

    sol3_proj = Counter(p for p, _ in sol3_bugs)
    togll_proj = Counter(p for p, _ in togll_bugs)
    toga_proj = Counter(p for p, _ in toga_bugs)

    for p in PROJECTS:
        s = sol3_proj.get(p, 0)
        t = togll_proj.get(p, 0)
        g = toga_proj.get(p, 0)
        diff = s - t
        print(f"  {p:20s} {s:5d} {t:6d} {g:5d} {diff:+6d}")

    s_total = sum(sol3_proj.values())
    t_total = sum(togll_proj.values())
    g_total = sum(toga_proj.values())
    print(f"  {'-'*45}")
    print(f"  {'Total':20s} {s_total:5d} {t_total:6d} {g_total:5d} {s_total-t_total:+6d}")


if __name__ == '__main__':
    main()

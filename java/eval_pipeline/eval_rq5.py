"""
RQ5 Evaluation: Bug Detection on Defects4J
Đánh giá khả năng phát hiện bugs thực tế
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import pandas as pd

import config
import utils
from oracle_generator import OracleGenerator


def eval_rq5(
    model_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    subset_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate RQ5: Bug detection on Defects4J.
    
    Args:
        model_path: Path to fine-tuned model
        output_dir: Directory to save results
        subset_size: Limit evaluation to N samples
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("RQ5 EVALUATION: Bug Detection (Defects4J)")
    print("="*60 + "\n")
    
    # Load oracle generator
    print("Loading model...")
    if model_path is None:
        default_path = Path(__file__).parent.parent / \
                     'RQ2/inference/run_inference/codeparrot_tc_mut_doc'
        model_path = str(default_path)
    
    generator = OracleGenerator(model_path)
    
    # Load Defects4J dataset
    print("Loading Defects4J dataset...")
    
    if not config.RQ5_INPUTS.exists():
        print(f"Error: Dataset not found at {config.RQ5_INPUTS}")
        return {}
    
    inputs_df = pd.read_csv(config.RQ5_INPUTS)
    meta_df = pd.read_csv(config.RQ5_META)
    
    print(f"Loaded {len(inputs_df)} test cases from Defects4J")
    
    if subset_size:
        inputs_df = inputs_df.head(subset_size)
        meta_df = meta_df.head(subset_size)
        print(f"Using subset: {subset_size} samples")
    
    # Step 1: Generate oracles
    print("\nStep 1: Generating oracles...")
    
    predictions = []
    oracle_types = []
    ground_truth_types = []
    
    start_time = time.time()
    
    for idx in tqdm(range(len(inputs_df)), desc="Generating"):
        input_row = inputs_df.iloc[idx]
        meta_row = meta_df.iloc[idx]
        
        test_prefix = input_row['test_prefix']
        focal_method = input_row['focal_method'] if not pd.isna(input_row['focal_method']) else None
        docstring = input_row['docstring'] if not pd.isna(input_row['docstring']) else None
        
        # Ground truth from meta.csv
        exception_bug = int(meta_row['exception_bug'])
        ground_truth_types.append(1 if exception_bug else 0)
        
        try:
            pred_oracle = generator.generate(test_prefix, focal_method, docstring)
            
            # Classify predicted oracle type
            if pred_oracle.strip().lower() == "exception":
                oracle_types.append(1)
            else:
                oracle_types.append(0)
            
            predictions.append({
                'id': idx,
                'project': meta_row['project'],
                'bug_num': meta_row['bug_num'],
                'test_name': meta_row['test_name'],
                'test_prefix': test_prefix[:100] + '...',
                'predicted_oracle': pred_oracle,
                'oracle_type': oracle_types[-1],
                'ground_truth_type': ground_truth_types[-1],
                'exception_bug': exception_bug,
                'assertion_bug': int(meta_row['assertion_bug'])
            })
            
        except Exception as e:
            print(f"\nError for sample {idx}: {e}")
            oracle_types.append(0)
            predictions.append({
                'id': idx,
                'predicted_oracle': "",
                'oracle_type': 0,
                'ground_truth_type': ground_truth_types[-1]
            })
    
    generation_time = time.time() - start_time
    
    # Step 2: Evaluate oracle classification (Table IV)
    print("\nStep 2: Computing oracle classification metrics...")
    
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    
    classification_metrics = {
        'overall_precision': float(precision_score(ground_truth_types, oracle_types, average='macro', zero_division=0)),  # type: ignore
        'overall_recall': float(recall_score(ground_truth_types, oracle_types, average='macro', zero_division=0)),  # type: ignore
        'overall_f1': float(f1_score(ground_truth_types, oracle_types, average='macro', zero_division=0)),  # type: ignore
        'assertion_precision': float(precision_score(ground_truth_types, oracle_types, pos_label=0, zero_division=0)),  # type: ignore
        'assertion_recall': float(recall_score(ground_truth_types, oracle_types, pos_label=0, zero_division=0)),  # type: ignore
        'exception_precision': float(precision_score(ground_truth_types, oracle_types, pos_label=1, zero_division=0)),  # type: ignore
        'exception_recall': float(recall_score(ground_truth_types, oracle_types, pos_label=1, zero_division=0))  # type: ignore
    }
    
    print(f"\nTable IV: Oracle Classification Metrics")
    print(f"  Overall Precision: {classification_metrics['overall_precision']:.4f}")
    print(f"  Overall Recall: {classification_metrics['overall_recall']:.4f}")
    print(f"  Assertion P/R: {classification_metrics['assertion_precision']:.4f} / {classification_metrics['assertion_recall']:.4f}")
    print(f"  Exception P/R: {classification_metrics['exception_precision']:.4f} / {classification_metrics['exception_recall']:.4f}")
    
    # Step 3: Split predictions by oracle type
    print("\nStep 3: Splitting predictions by oracle type...")
    
    num_assertion = sum(1 for t in oracle_types if t == 0)
    num_exception = sum(1 for t in oracle_types if t == 1)
    
    print(f"  Assertion oracles: {num_assertion}")
    print(f"  Exception oracles: {num_exception}")
    
    # Step 4: Bug detection info
    print("\nStep 4: Bug detection analysis...")
    print("Note: Full bug detection requires Docker execution")
    print("      See RQ5/README.txt for instructions")
    
    bug_detection_metrics = {
        'total_tests': len(predictions),
        'assertion_oracles': num_assertion,
        'exception_oracles': num_exception,
        'generation_time_seconds': generation_time,
        'note': 'Run tests in Docker to get TP/FP/FN metrics'
    }
    
    all_metrics = {
        'classification': classification_metrics,
        'bug_detection': bug_detection_metrics,
        'generation_time_seconds': generation_time
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all predictions
        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv(output_dir / "oracle_predictions.csv", index=False)
        
        # Split by oracle type (for Docker execution)
        assertion_preds = df_predictions[df_predictions['oracle_type'] == 0]
        exception_preds = df_predictions[df_predictions['oracle_type'] == 1]
        
        assertion_dir = output_dir / 'assertion_prefix'
        exception_dir = output_dir / 'exception_prefix'
        prefix_only_dir = output_dir / 'prefix_only'
        
        assertion_dir.mkdir(exist_ok=True)
        exception_dir.mkdir(exist_ok=True)
        prefix_only_dir.mkdir(exist_ok=True)
        
        assertion_preds.to_csv(assertion_dir / 'oracle_preds.csv', index=False)
        exception_preds.to_csv(exception_dir / 'oracle_preds.csv', index=False)
        df_predictions[['id', 'project', 'bug_num', 'test_name']].to_csv(
            prefix_only_dir / 'oracle_preds.csv', index=False
        )
        
        # Save metrics
        utils.save_json(all_metrics, output_dir / "rq5_metrics.json")
        
        # Save classification report
        report = classification_report(
            ground_truth_types,
            oracle_types,
            target_names=['Assertion', 'Exception'],
            output_dict=True
        )
        utils.save_json(report, output_dir / "classification_report.json")
        
        print(f"\nResults saved to: {output_dir}")
        print("\nTo complete bug detection (Table V):")
        print("1. Use oracle_preds.csv files in assertion_prefix/ and exception_prefix/")
        print("2. Copy to Docker: docker cp assertion_prefix/oracle_preds.csv toga:/home/...")
        print("3. Run in Docker: bash rq3.sh")
        print("4. Copy results back: docker cp toga:/home/.../togll_generated .")
        print("5. Run analysis: python analyze_bug_detection.py")
    
    return all_metrics


def analyze_bug_detection_results(
    togll_results_dir: Path,
    toga_results_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Analyze bug detection results from Docker test execution.
    Computes Table V metrics.
    
    Args:
        togll_results_dir: Path to TOGLL results (e.g., RQ5/bug_detection/TOGLL)
        toga_results_dir: Path to TOGA results for comparison (optional)
        
    Returns:
        metrics: Bug detection metrics
    """
    print("\nAnalyzing bug detection results...")
    
    metrics: Dict[str, Any] = {}
    
    if togll_results_dir.exists():
        togll_metrics = _analyze_single_method(togll_results_dir, "TOGLL")
        metrics['togll'] = togll_metrics
    
    if toga_results_dir and toga_results_dir.exists():
        toga_metrics = _analyze_single_method(toga_results_dir, "TOGA")
        metrics['toga'] = toga_metrics
        
        # Compute unique bugs
        if 'togll' in metrics:
            togll_bugs = metrics['togll']['unique_bugs']
            toga_bugs = toga_metrics['unique_bugs']
            
            metrics['comparison'] = {
                'toga_unique': len(toga_bugs - togll_bugs),
                'togll_unique': len(togll_bugs - toga_bugs),
                'overlap': len(toga_bugs & togll_bugs),
                'total_toga': len(toga_bugs),
                'total_togll': len(togll_bugs)
            }
            
            print("\nTOGA vs TOGLL Comparison:")
            print(f"  TOGA unique bugs: {metrics['comparison']['toga_unique']}")
            print(f"  TOGLL unique bugs: {metrics['comparison']['togll_unique']}")
            print(f"  Overlap: {metrics['comparison']['overlap']}")
    
    return metrics


def _analyze_single_method(results_dir: Path, method_name: str) -> Dict[str, Any]:
    """Analyze results for a single method (TOGA or TOGLL)."""
    
    bugs_found = set()
    
    for oracle_type in ['assertion_prefix', 'exception_prefix', 'prefix_only']:
        test_data_file = results_dir / oracle_type / f"{method_name.lower()}_generated" / "test_data.csv"
        
        if test_data_file.exists():
            df = pd.read_csv(test_data_file)
            filtered = df[df['TP'] == True]
            if len(filtered) > 0:
                bugs_list = filtered[['project', 'bug_num']].values.tolist()  # type: ignore
                seen = set()
                for bug_info in bugs_list:
                    bug_tuple = (bug_info[0], bug_info[1])
                    if bug_tuple not in seen:
                        seen.add(bug_tuple)
                        bugs_found.add(bug_tuple)
    
    print(f"\n{method_name}: {len(bugs_found)} unique bugs detected")
    
    return {
        'unique_bugs': bugs_found,
        'total_bugs_detected': len(bugs_found)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate RQ5 - Bug Detection')
    parser.add_argument(
        '--model',
        type=str,
        help='Path to fine-tuned model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/rq5',
        help='Output directory'
    )
    parser.add_argument(
        '--subset',
        type=int,
        help='Limit to N samples'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze existing test results'
    )
    parser.add_argument(
        '--togll-results',
        type=str,
        help='Path to TOGLL test execution results'
    )
    parser.add_argument(
        '--toga-results',
        type=str,
        help='Path to TOGA test execution results'
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        if not args.togll_results:
            print("Error: --togll-results required for analysis")
            return
        
        toga_path = Path(args.toga_results) if args.toga_results else None
        metrics = analyze_bug_detection_results(
            Path(args.togll_results),
            toga_path
        )
    else:
        metrics = eval_rq5(
            model_path=args.model,
            output_dir=Path(args.output),
            subset_size=args.subset
        )
    
    print("\nRQ5 Evaluation completed!")


if __name__ == '__main__':
    main()


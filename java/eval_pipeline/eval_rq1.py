import argparse
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

import config
import utils
from oracle_generator import OracleGenerator


def load_rq1_data_from_csv(inputs_csv, meta_csv, subset_size=None):
    """
    Load RQ1 data from CSV files instead of pickle.
    More general and easier to experiment with different methods.
    
    Returns:
        List of dicts with 'test_prefix', 'focal_method', 'docstring', 'target_oracle'
    """
    print(f"Loading inputs from {inputs_csv}...")
    inputs_df = pd.read_csv(inputs_csv)
    
    print(f"Loading metadata from {meta_csv}...")
    meta_df = pd.read_csv(meta_csv)
    
    # Merge on id
    merged = pd.merge(inputs_df, meta_df, on='id', how='inner')
    
    print(f"Total samples: {len(merged)}")
    
    # Apply subset if specified
    if subset_size:
        merged = merged.head(subset_size)
        print(f"Using subset: {subset_size} samples")
    
    # Convert to list of dicts
    data = []
    for _, row in merged.iterrows():
        # Extract ground truth oracle from GTassert column
        gt_oracle = row.get('GTassert', '')
        if pd.isna(gt_oracle):
            gt_oracle = ''
        
        # Check if it's exception test
        assert_or_exception = row.get('assertORexception', 0)
        if assert_or_exception == -1:
            gt_oracle = 'exception'
        
        data.append({
            'id': row['id'],
            'test_prefix': row['test_prefix'],
            'focal_method': None if pd.isna(row['focal_method']) else row['focal_method'],
            'docstring': None if pd.isna(row['docstring']) else row['docstring'],
            'target_oracle': gt_oracle.strip() if gt_oracle else ''
        })
    
    return data


def eval_rq1(
    model_path: str,
    output_dir: Path,
    subset_size: int,
    inputs_csv: str,
    meta_csv: str
) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("RQ1 EVALUATION: Intrinsic Accuracy")
    print("="*60 + "\n")
    
    generator = OracleGenerator(str(Path(model_path)))
    
    # Load dataset from CSV
    if inputs_csv is None:
        inputs_csv = config.RQ1_INPUTS
    if meta_csv is None:
        meta_csv = config.RQ1_META
    
    if not Path(inputs_csv).exists():
        print(f"Error: inputs.csv not found at {inputs_csv}")
        return {}
    
    if not Path(meta_csv).exists():
        print(f"Error: meta.csv not found at {meta_csv}")
        return {}
    
    eval_data = load_rq1_data_from_csv(inputs_csv, meta_csv, subset_size)
    
    if not eval_data:
        print("No data to evaluate")
        return {}
    
    # Generate predictions
    print("\nGenerating oracles...")
    predictions = []
    targets = []
    
    start_time = time.time()
    
    for sample in tqdm(eval_data, desc="Generating"):
        try:
            pred_oracle = generator.generate(
                test_prefix=sample['test_prefix'],
                focal_method=sample['focal_method'],
                docstring=sample['docstring']
            )
            
            predictions.append(pred_oracle)
            targets.append(sample['target_oracle'])
            
        except Exception as e:
            print(f"\nError generating oracle for sample {sample['id']}: {e}")
            predictions.append("")
            targets.append(sample['target_oracle'])
    
    generation_time = time.time() - start_time
    
    # Compute metrics
    print("\nComputing metrics...")
    
    exact_match_acc = utils.compute_exact_match(predictions, targets)
    
    metrics = {
        'dataset': 'SF110',
        'config': 'tc_mut_doc',
        'num_samples': len(predictions),
        'exact_match_accuracy': exact_match_acc,
        'generation_time_seconds': generation_time,
        'avg_time_per_sample': generation_time / len(predictions) if predictions else 0
    }
    
    utils.print_metrics(metrics, "RQ1 RESULTS")
    
    # Save detailed results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, (sample, pred, target) in enumerate(zip(eval_data, predictions, targets)):
            results.append({
                'id': sample['id'],
                'test_prefix': sample['test_prefix'][:100] + '...',  # Truncate for readability
                'prediction': pred,
                'target': target,
                'exact_match': pred.strip() == target.strip()
            })
        
        utils.save_json(results, output_dir / "rq1_predictions.json")
        utils.save_json(metrics, output_dir / "rq1_metrics.json")
        
        print(f"\nResults saved to: {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate RQ1 - CSV based (general)')
    parser.add_argument(
        '--model',
        type=str,
        help='Path to fine-tuned model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/rq1',
        help='Output directory for results'
    )
    parser.add_argument(
        '--subset',
        type=int,
        help='Limit to N samples (for quick testing)'
    )
    parser.add_argument(
        '--inputs',
        type=str,
        help='Path to inputs.csv'
    )
    parser.add_argument(
        '--meta',
        type=str,
        help='Path to meta.csv'
    )
    
    args = parser.parse_args()
    
    metrics = eval_rq1(
        model_path=args.model,
        output_dir=Path(args.output),
        subset_size=args.subset,
        inputs_csv=args.inputs,
        meta_csv=args.meta
    )
    
    print("\nEvaluation completed!")
    print(f"Exact Match Accuracy: {metrics.get('exact_match_accuracy', 0):.4f}")


if __name__ == '__main__':
    main()
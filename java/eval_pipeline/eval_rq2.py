"""
RQ2 Evaluation: Generalization to New Projects
Đánh giá khả năng generalize sang 25 Apache Commons projects
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import pandas as pd

import config
import utils
from oracle_generator import OracleGenerator


def load_rq2_project_data(project_path: Path, subset_size: int = None):
    """
    Load RQ2 project data from CSV files.
    
    Returns:
        List of dicts with 'test_prefix', 'focal_method', 'docstring', 'target_oracle'
    """
    inputs_file = project_path / "inputs.csv"
    meta_file = project_path / "meta_llm.csv"
    
    if not inputs_file.exists():
        raise FileNotFoundError(f"inputs.csv not found at {inputs_file}")
    
    if not meta_file.exists():
        raise FileNotFoundError(f"meta_llm.csv not found at {meta_file}")
    
    inputs_df = pd.read_csv(inputs_file)
    meta_df = pd.read_csv(meta_file)
    
    if subset_size:
        inputs_df = inputs_df.head(subset_size)
        meta_df = meta_df.head(subset_size)
    
    data = []
    for idx in range(min(len(inputs_df), len(meta_df))):
        input_row = inputs_df.iloc[idx]
        meta_row = meta_df.iloc[idx]
        
        data.append({
            'id': meta_row.get('id', idx),
            'test_prefix': input_row['test_prefix'],
            'focal_method': input_row['focal_method'] if not pd.isna(input_row['focal_method']) else None,
            'docstring': input_row['docstring'] if not pd.isna(input_row['docstring']) else None,
            'target_oracle': meta_row['GT_output'].strip() if not pd.isna(meta_row['GT_output']) else ''
        })
    
    return data


def eval_rq2(
    model_path: str = None,
    projects: List[str] = None,
    output_dir: Path = None,
    subset_size: int = None
) -> Dict[str, Any]:
    """
    Evaluate RQ2: Generalization on Apache Commons projects.
    
    Args:
        model_path: Path to fine-tuned model
        projects: List of project names to evaluate
        output_dir: Directory to save results
        subset_size: Limit evaluation to N samples per project
        
    Returns:
        metrics: Dictionary of evaluation metrics per project
    """
    print("\n" + "="*60)
    print("RQ2 EVALUATION: Generalization")
    print("="*60 + "\n")
    
    # Load oracle generator
    print("Loading model...")
    if model_path is None:
        model_path = Path(__file__).parent.parent / \
                     'RQ2/inference/run_inference/codeparrot_tc_mut_doc'
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return {}
    
    print(f"Loading model from: {model_path}")
    generator = OracleGenerator(str(model_path))
    
    # Use all projects if not specified
    if projects is None:
        projects = config.PROJECTS
    
    print(f"Evaluating {len(projects)} projects...")
    
    overall_metrics = {
        'total_projects': 0,
        'total_samples': 0,
        'total_exact_match': 0,
        'per_project': {}
    }
    
    for project_name in projects:
        print(f"\n{'='*60}")
        print(f"Project: {project_name}")
        print(f"{'='*60}")
        
        project_path = config.RQ2_INFERENCE_DATA / project_name
        
        if not project_path.exists():
            print(f"Warning: Project not found at {project_path}, skipping...")
            continue
        
        try:
            # Load project data
            print(f"Loading data from {project_path}...")
            project_data = load_rq2_project_data(project_path, subset_size)
            print(f"Loaded {len(project_data)} test cases")
            
            if not project_data:
                print("No data to evaluate, skipping...")
                continue
            
            # Generate predictions
            print("Generating oracles...")
            predictions = []
            targets = []
            
            start_time = time.time()
            
            for sample in tqdm(project_data, desc="Generating"):
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
            
            # Compute metrics for this project
            exact_match_acc = utils.compute_exact_match(predictions, targets)
            
            project_metrics = {
                'num_samples': len(predictions),
                'exact_match': exact_match_acc,
                'generation_time_seconds': generation_time,
                'avg_time_per_sample': generation_time / len(predictions) if predictions else 0
            }
            
            overall_metrics['per_project'][project_name] = project_metrics
            overall_metrics['total_projects'] += 1
            overall_metrics['total_samples'] += len(predictions)
            overall_metrics['total_exact_match'] += sum(
                1 for p, t in zip(predictions, targets) if p.strip() == t.strip()
            )
            
            print(f"\n{project_name} Results:")
            print(f"  Samples: {len(predictions)}")
            print(f"  Exact Match: {exact_match_acc:.4f}")
            print(f"  Time: {utils.format_time(generation_time)}")
            
            # Save project-specific results
            if output_dir:
                project_output = Path(output_dir) / project_name
                project_output.mkdir(parents=True, exist_ok=True)
                
                results = []
                for sample, pred, target in zip(project_data, predictions, targets):
                    results.append({
                        'id': int(sample['id']),
                        'test_prefix': sample['test_prefix'][:100] + '...',
                        'prediction': pred,
                        'target': target,
                        'exact_match': pred.strip() == target.strip()
                    })
                
                utils.save_json(results, project_output / "predictions.json")
                utils.save_json(project_metrics, project_output / "metrics.json")
        
        except Exception as e:
            print(f"Error processing project {project_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute overall metrics
    if overall_metrics['total_samples'] > 0:
        overall_metrics['overall_exact_match'] = (
            overall_metrics['total_exact_match'] / overall_metrics['total_samples']
        )
    else:
        overall_metrics['overall_exact_match'] = 0.0
    
    utils.print_metrics(overall_metrics, "RQ2 OVERALL RESULTS")
    
    # Save overall results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        utils.save_json(overall_metrics, output_dir / "rq2_overall_metrics.json")
        
        # Create summary table
        summary = []
        for project, metrics in overall_metrics['per_project'].items():
            summary.append({
                'project': project,
                'samples': metrics['num_samples'],
                'accuracy': metrics['exact_match'],
                'time_seconds': metrics['generation_time_seconds']
            })
        
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(output_dir / "rq2_summary.csv", index=False)
        print(f"\nResults saved to: {output_dir}")
    
    return overall_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate RQ2 - Generalization')
    parser.add_argument(
        '--model',
        type=str,
        help='Path to fine-tuned model'
    )
    parser.add_argument(
        '--projects',
        nargs='+',
        help='List of projects to evaluate (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/rq2',
        help='Output directory for results'
    )
    parser.add_argument(
        '--subset',
        type=int,
        help='Limit to N samples per project'
    )
    
    args = parser.parse_args()
    
    metrics = eval_rq2(
        model_path=args.model,
        projects=args.projects,
        output_dir=Path(args.output),
        subset_size=args.subset
    )
    
    print("\nRQ2 Evaluation completed!")
    print(f"Overall Exact Match: {metrics.get('overall_exact_match', 0):.4f}")


if __name__ == '__main__':
    main()


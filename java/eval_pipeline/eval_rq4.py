"""
RQ4 Evaluation: Test Execution and Mutation Testing
Chạy tests thật với Maven và PITest
Tests must already have oracles injected via inject_oracles.py
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

import config
import utils


def eval_rq4(
    projects: List[str] = None,
    output_dir: Path = None,
    run_mutation: bool = False,
    test_versions: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate RQ4: Test execution and mutation testing.
    Requires oracles already injected in artifacts_with_es_togll_tests.
    
    Args:
        projects: List of projects to evaluate
        output_dir: Directory to save results
        run_mutation: Whether to run mutation testing (slow!)
        test_versions: Which versions to test ['evosuite', 'togll', 'no_oracle']
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("RQ4 EVALUATION: Test Execution + Mutation Testing")
    print("="*60 + "\n")
    
    if test_versions is None:
        test_versions = ['togll']  # Default to TOGLL version
    
    if projects is None:
        # Use subset for quick testing
        projects = config.PROJECTS[:3]
        print(f"Using subset of 3 projects for testing")
    
    print(f"Projects to evaluate: {len(projects)}")
    print(f"Test versions: {test_versions}")
    print(f"Run mutation testing: {run_mutation}")
    
    all_metrics = {
        'projects_tested': 0,
        'version_metrics': {}
    }
    
    for version in test_versions:
        print(f"\n{'='*60}")
        print(f"Testing version: {version.upper()}")
        print(f"{'='*60}")
        
        version_metrics = {
            'version': version,
            'projects': {},
            'overall': {
                'total_tests': 0,
                'total_passed': 0,
                'total_failed': 0,
                'total_errors': 0,
                'compilation_success': 0,
                'compilation_total': 0
            }
        }
        
        for project_name in projects:
            print(f"\n{'='*60}")
            print(f"Project: {project_name} ({version})")
            print(f"{'='*60}")
            
            project_path = config.RQ4_PROJECTS / project_name
            
            if not project_path.exists():
                print(f"Warning: Project not found, skipping...")
                continue
            
            # Determine test directory based on version
            if version == 'evosuite':
                test_dir = 'src'
                target_dir = 'target'
            elif version == 'togll':
                test_dir = 'llm_oracle'
                target_dir = 'llm_oracle/target'
            elif version == 'no_oracle':
                test_dir = 'no_oracle'
                target_dir = 'no_oracle/target'
            else:
                print(f"Unknown version: {version}")
                continue
            
            # Step 1: Clean and compile
            print("Step 1: Compiling...")
            compile_result = utils.run_maven_command(
                project_path,
                f"clean compile -Dtest.dir={test_dir}",
                timeout=600
            )
            
            project_metrics = {
                'compilation_success': compile_result['success'],
                'compilation_time': 0
            }
            
            if not compile_result['success']:
                print(f"✗ Compilation failed!")
                version_metrics['projects'][project_name] = project_metrics
                continue
            
            print(f"✓ Compilation succeeded")
            version_metrics['overall']['compilation_success'] += 1
            
            # Step 2: Run tests
            print("\nStep 2: Running tests...")
            test_result = utils.run_maven_command(
                project_path,
                f"test -Dtest.dir={test_dir} -Dtarget.dir={target_dir}",
                timeout=1800,
                log_file=Path(output_dir) / project_name / f"{version}_test.log" if output_dir else None
            )
            
            # Parse test results
            test_stats = utils.parse_maven_test_output(test_result['stdout'])
            
            project_metrics.update({
                'tests_run': test_stats['tests'],
                'tests_passed': test_stats['tests'] - test_stats['failures'] - test_stats['errors'],
                'tests_failed': test_stats['failures'],
                'tests_errors': test_stats['errors'],
                'tests_skipped': test_stats['skipped']
            })
            
            if test_stats['tests'] > 0:
                project_metrics['pass_rate'] = (
                    project_metrics['tests_passed'] / test_stats['tests']
                )
            else:
                project_metrics['pass_rate'] = 0.0
            
            print(f"Tests run: {test_stats['tests']}")
            print(f"Passed: {project_metrics['tests_passed']}")
            print(f"Failed: {test_stats['failures']}")
            print(f"Errors: {test_stats['errors']}")
            print(f"Pass rate: {project_metrics['pass_rate']:.2%}")
            
            # Update overall stats
            version_metrics['overall']['total_tests'] += test_stats['tests']
            version_metrics['overall']['total_passed'] += project_metrics['tests_passed']
            version_metrics['overall']['total_failed'] += test_stats['failures']
            version_metrics['overall']['total_errors'] += test_stats['errors']
            
            # Step 3: Mutation testing (optional)
            if run_mutation:
                print("\nStep 3: Running mutation testing (this may take a while)...")
                
                mutation_result = utils.run_maven_command(
                    project_path,
                    f"org.pitest:pitest-maven:mutationCoverage -Dtest.dir={test_dir} -Dtarget.dir={target_dir}",
                    timeout=7200,  # 2 hours
                    log_file=Path(output_dir) / project_name / f"{version}_mutation.log" if output_dir else None
                )
                
                if mutation_result['success']:
                    # Parse mutation testing results
                    pit_output = Path(output_dir) / project_name / f"{version}_mutation.log"
                    if pit_output.exists():
                        mutation_metrics = utils.parse_pitest_output(pit_output)
                        project_metrics['mutation'] = mutation_metrics
                        print(f"Mutation coverage: {mutation_metrics.get('mutation_coverage', 0):.2f}%")
                else:
                    print("✗ Mutation testing failed")
            
            version_metrics['projects'][project_name] = project_metrics
            
            # Save project results
            if output_dir:
                project_output = Path(output_dir) / project_name
                project_output.mkdir(parents=True, exist_ok=True)
                utils.save_json(
                    project_metrics,
                    project_output / f"{version}_metrics.json"
                )
        
        # Compute overall pass rate
        if version_metrics['overall']['total_tests'] > 0:
            version_metrics['overall']['overall_pass_rate'] = (
                version_metrics['overall']['total_passed'] / 
                version_metrics['overall']['total_tests']
            )
        else:
            version_metrics['overall']['overall_pass_rate'] = 0.0
        
        version_metrics['overall']['compilation_rate'] = (
            version_metrics['overall']['compilation_success'] / len(projects)
        )
        
        all_metrics['version_metrics'][version] = version_metrics
        
        utils.print_metrics(version_metrics['overall'], f"{version.upper()} OVERALL RESULTS")
    
    # Save overall results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        utils.save_json(all_metrics, output_dir / "rq4_overall_metrics.json")
        
        # Create comparison table
        import pandas as pd
        comparison = []
        for version, metrics in all_metrics['version_metrics'].items():
            comparison.append({
                'version': version,
                'compilation_rate': metrics['overall']['compilation_rate'],
                'total_tests': metrics['overall']['total_tests'],
                'pass_rate': metrics['overall']['overall_pass_rate']
            })
        
        df = pd.DataFrame(comparison)
        df.to_csv(output_dir / "rq4_comparison.csv", index=False)
        
        print(f"\nResults saved to: {output_dir}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate RQ4 - Test Execution')
    parser.add_argument(
        '--projects',
        nargs='+',
        help='List of projects to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/rq4',
        help='Output directory'
    )
    parser.add_argument(
        '--mutation',
        action='store_true',
        help='Run mutation testing (slow!)'
    )
    parser.add_argument(
        '--versions',
        nargs='+',
        choices=['evosuite', 'togll', 'no_oracle'],
        default=['togll'],
        help='Test versions to evaluate'
    )
    
    args = parser.parse_args()
    
    metrics = eval_rq4(
        projects=args.projects,
        output_dir=Path(args.output),
        run_mutation=args.mutation,
        test_versions=args.versions
    )
    
    print("\nRQ4 Evaluation completed!")
    
    for version, v_metrics in metrics['version_metrics'].items():
        overall = v_metrics['overall']
        print(f"\n{version.upper()}:")
        print(f"  Compilation rate: {overall['compilation_rate']:.2%}")
        print(f"  Pass rate: {overall['overall_pass_rate']:.2%}")


if __name__ == '__main__':
    main()


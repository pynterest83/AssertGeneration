"""
Utility functions for evaluation pipeline
"""

import subprocess
import json
import csv
import pickle
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


def load_pickle(path: Path) -> Any:
    """Load pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV file."""
    return pd.read_csv(path)


def save_json(data: Any, path: Path):
    """Save data to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def compute_exact_match(predictions: List[str], targets: List[str]) -> float:
    """Compute exact match accuracy."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    matches = sum(1 for pred, target in zip(predictions, targets) 
                  if pred.strip() == target.strip())
    return matches / len(predictions)


def run_maven_command(
    project_path: Path,
    command: str,
    timeout: int = 3600,
    log_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run Maven command in a Java project.
    
    Args:
        project_path: Path to Java project with pom.xml
        command: Maven command (e.g., "clean test")
        timeout: Command timeout in seconds
        log_file: Optional path to save logs
        
    Returns:
        result: Dict with returncode, stdout, stderr
    """
    if not (project_path / "pom.xml").exists():
        raise FileNotFoundError(f"pom.xml not found in {project_path}")
    
    cmd = f"mvn {command} -Devosuite.skip=true -Dmaven.test.skip=false -pl client"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
        # Save logs if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'w') as f:
                f.write(f"Command: {cmd}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"\nSTDOUT:\n{result.stdout}\n")
                f.write(f"\nSTDERR:\n{result.stderr}\n")
        
        return output
        
    except subprocess.TimeoutExpired:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': f'Command timeout after {timeout}s',
            'success': False
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }


def parse_maven_test_output(stdout: str) -> Dict[str, int]:
    """
    Parse Maven test output to extract test statistics.
    
    Returns:
        stats: Dict with tests, failures, errors, skipped
    """
    stats = {
        'tests': 0,
        'failures': 0,
        'errors': 0,
        'skipped': 0
    }
    
    # Pattern: Tests run: 123, Failures: 4, Errors: 2, Skipped: 1
    pattern = r'Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)'
    matches = re.findall(pattern, stdout)
    
    if matches:
        # Take the last match (summary line)
        tests, failures, errors, skipped = matches[-1]
        stats = {
            'tests': int(tests),
            'failures': int(failures),
            'errors': int(errors),
            'skipped': int(skipped)
        }
    
    return stats


def parse_pitest_output(output_file: Path) -> Dict[str, Any]:
    """
    Parse PITest mutation testing output.
    
    Returns:
        metrics: Dict with mutation coverage, line coverage, etc.
    """
    if not output_file.exists():
        return {}
    
    with open(output_file) as f:
        content = f.read()
    
    metrics = {}
    
    # Extract metrics using regex
    patterns = {
        'line_coverage': r'Line Coverage:\s*([\d.]+)%',
        'mutation_coverage': r'Mutation Coverage:\s*([\d.]+)%',
        'test_strength': r'Test Strength:\s*([\d.]+)%',
        'mutations_generated': r'Generated (\d+) mutations',
        'mutations_killed': r'Killed (\d+)',
        'mutations_survived': r'Survived (\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = int(value)
    
    return metrics


def inject_oracle_into_test(
    test_file: Path,
    test_id: int,
    oracle: str,
    backup: bool = True
) -> bool:
    """
    Inject oracle into a Java test file.
    
    Args:
        test_file: Path to Java test file
        test_id: Test method ID
        oracle: Oracle statement to inject
        backup: Whether to backup original file
        
    Returns:
        success: Whether injection succeeded
    """
    try:
        # Read original content
        with open(test_file) as f:
            content = f.read()
        
        # Backup if requested
        if backup:
            backup_file = test_file.with_suffix('.java.bak')
            with open(backup_file, 'w') as f:
                f.write(content)
        
        # Find test method and inject oracle
        # This is simplified - actual implementation needs more robust parsing
        pattern = rf'(public void test{test_id}\(\).*?\{{[^}}]*?)(\}})'
        
        def replace_func(match):
            method_body = match.group(1)
            closing_brace = match.group(2)
            # Add oracle before closing brace
            return f"{method_body}\n    {oracle}\n{closing_brace}"
        
        modified_content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
        
        # Write modified content
        with open(test_file, 'w') as f:
            f.write(modified_content)
        
        return True
        
    except Exception as e:
        print(f"Error injecting oracle: {e}")
        return False


def clean_test_prefix(test_prefix: str) -> str:
    """
    Clean test prefix by removing existing oracles.
    
    Args:
        test_prefix: Original test code
        
    Returns:
        cleaned: Test code without oracles
    """
    # Remove try-catch blocks
    cleaned = re.sub(r'try\s*\{', '', test_prefix)
    cleaned = re.sub(r'fail\s*\([^)]*\);', '', cleaned)
    cleaned = re.sub(r'\}\s*catch\s*\([^)]*\)\s*\{.*?\}', '', cleaned, flags=re.DOTALL)
    
    # Remove assertions
    assert_re = re.compile(r'assert\w*\(.*?\);', re.DOTALL)
    cleaned = re.sub(assert_re, '', cleaned)
    
    # Remove empty lines
    lines = [line for line in cleaned.split('\n') if line.strip()]
    cleaned = '\n'.join(lines)
    
    return cleaned


def is_exception_oracle(test_prefix: str) -> bool:
    """Check if test expects an exception."""
    fail_catch_pattern = re.compile(r"fail\(.*\).*}\s*catch", re.MULTILINE | re.DOTALL)
    return bool(re.search(fail_catch_pattern, test_prefix))


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_metrics(metrics: Dict[str, Any], title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<50} {value:.4f}")
        else:
            print(f"  {key:.<50} {value}")
    
    print(f"{'='*60}\n")


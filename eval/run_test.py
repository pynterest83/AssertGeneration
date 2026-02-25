import re
import subprocess
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def run_maven_test(project_path: Path) -> str:
    cmd = "mvn test -B -Drat.skip=true"
    result = subprocess.run(cmd, shell=True, cwd=str(project_path), capture_output=True, text=True)
    log = result.stdout + "\n" + result.stderr
    
    log_file = project_path / "test_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(log)
    
    return log


def count_test_failures(project_path: Path) -> int:
    """Count Tfp from surefire reports"""
    test_regex = re.compile(r"Tests run:\s*(\d+).*Failures:\s*(\d+).*Errors:\s*(\d+)")
    total_failures = 0
    
    for surefire_dir in project_path.rglob("**/surefire-reports"):
        if not surefire_dir.is_dir():
            continue
        for txt_file in surefire_dir.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = test_regex.search(line)
                    if match:
                        total_failures += int(match.group(2)) + int(match.group(3))
    return total_failures


def count_total_tests_from_meta(meta_csv: str) -> int:
    """Count T from meta_llm.csv (only assertion tests, exclude exception)"""
    df = pd.read_csv(meta_csv)
    # Filter only assertion tests (GT_output != 'exception')
    assertion_tests = df[df['GT_output'] != 'exception']
    return len(assertion_tests)


def load_compile_results(input_dir: Path) -> int:
    """Load Tce from compile_results.json"""
    results_file = input_dir / "compile_results.json"
    if not results_file.exists():
        print(f"Warning: {results_file} not found. Run run_compile.py first.")
        return 0
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data.get('total_Tce', 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--meta_csv', type=str, required=True, help='Path to meta_llm.csv')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--projects', type=str, nargs='*', default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Get T from meta_csv
    T = count_total_tests_from_meta(args.meta_csv)
    print(f"Total assertion tests (T): {T}")
    
    # Get Tce from compile_results.json
    Tce = load_compile_results(input_dir)
    print(f"Compilation errors (Tce): {Tce}")

    # Determine which projects to test
    if args.projects:
        projects = [input_dir / p for p in args.projects if (input_dir / p).is_dir()]
    else:
        projects = [p for p in input_dir.iterdir() if p.is_dir() and p.name not in ['results', 'compiled', 'infer_input', 'toga_output', '.github', 'travis']]

    # Run tests and count Tfp
    total_Tfp = 0
    project_results = {}
    
    for project_path in tqdm(sorted(projects), desc="Running tests"):
        print(f"\nTesting: {project_path.name}")
        run_maven_test(project_path)
        tfp = count_test_failures(project_path)
        total_Tfp += tfp
        project_results[project_path.name] = {'Tfp': tfp}
        print(f"  Failures: {tfp}")

    # Calculate SR
    SR = (T - Tce - total_Tfp) / T if T > 0 else 0.0

    # Save results
    output_file = Path(args.output) if args.output else input_dir / "test_results.json"
    results = {
        'T': T,
        'Tce': Tce,
        'Tfp': total_Tfp,
        'SR': SR,
        'projects': project_results
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*40}")
    print(f"T  (total assertion tests): {T}")
    print(f"Tce (compilation errors):   {Tce}")
    print(f"Tfp (test failures):        {total_Tfp}")
    print(f"SR  (success rate):         {SR:.4f} ({SR*100:.2f}%)")
    print(f"{'='*40}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()

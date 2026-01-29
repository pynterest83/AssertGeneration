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


def count_compilation_errors(project_path: Path) -> int:
    error_file = project_path / "compilation_error.txt"
    if not error_file.exists():
        return 0
    
    count = 0
    with open(error_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "[ERROR] " in line and ".java:[" in line:
                count += 1
    return count


def count_test_failures(project_path: Path) -> int:
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


def count_total_tests(project_path: Path) -> int:
    csv_file = project_path / "outputs" / "oracle_preds.csv"
    if not csv_file.exists():
        return 0
    
    try:
        df = pd.read_csv(csv_file)
        return len(df)
    except:
        return 0


def process_project(project_path: Path) -> dict:
    Tce = count_compilation_errors(project_path)
    run_maven_test(project_path)
    
    T = count_total_tests(project_path)
    Tfp = count_test_failures(project_path)
    SR = (T - Tce - Tfp) / T if T > 0 else 0.0
    
    return {
        'project': project_path.name,
        'total_tests': T,
        'compilation_errors': Tce,
        'false_positives': Tfp,
        'success_rate': SR
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--projects', type=str, nargs='*', default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if args.projects:
        projects = [input_dir / p for p in args.projects if (input_dir / p).is_dir()]
    else:
        projects = [p for p in input_dir.iterdir() if p.is_dir() and p.name != 'results']

    all_results = []
    for project_path in tqdm(sorted(projects), desc="Processing projects"):
        result = process_project(project_path)
        all_results.append(result)

    # Summary
    total_T = sum(r['total_tests'] for r in all_results)
    total_Tce = sum(r['compilation_errors'] for r in all_results)
    total_Tfp = sum(r['false_positives'] for r in all_results)
    overall_SR = (total_T - total_Tce - total_Tfp) / total_T if total_T > 0 else 0.0

    output_file = Path(args.output) if args.output else input_dir / "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_tests': total_T,
            'compilation_errors': total_Tce,
            'false_positives': total_Tfp,
            'success_rate': overall_SR,
            'details': all_results
        }, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()

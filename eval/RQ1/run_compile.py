import shutil
import subprocess
import argparse
import json
from pathlib import Path
from comment_incompatible_assertions import comment_assertions


def run_maven_compile(project_path: Path, timeout: int = 300) -> str:
    cmd = "mvn test-compile -fae -B -Drat.skip=true"
    result = subprocess.run(cmd, shell=True, cwd=str(project_path), timeout=timeout, capture_output=True, text=True)
    log = result.stdout + "\n" + result.stderr
    
    error_log = project_path / "compilation_error.txt"
    with open(error_log, 'w', encoding='utf-8') as f:
        f.write(log)
    
    return log


def process_project(project_path: Path, timeout: int = 300) -> tuple[bool, int]:
    print(f"\nProcessing: {project_path.name}")
    total_commented = 0
    
    while True:
        log = run_maven_compile(project_path, timeout)
        
        if 'BUILD SUCCESS' in log:
            print(f"  BUILD SUCCESS")
            return True, total_commented
        
        if '.java:[' not in log:
            print(f"  BUILD FAILED (no compilation errors)")
            return False, total_commented
        
        error_log = project_path / "compilation_error.txt"
        commented = comment_assertions(str(error_log))
        total_commented += commented
        print(f"  Commented {commented} lines")
        
        if commented == 0:
            return False, total_commented


def copy_project(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=lambda d, f: [x for x in f if x in ['target', '.git']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--projects', type=str, nargs='*', default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if args.projects:
        projects = []
        for p in args.projects:
            candidate = input_dir / p
            if candidate.is_dir() and (candidate / "pom.xml").exists():
                projects.append(candidate)
        if not projects and (input_dir / "pom.xml").exists():
            projects = [input_dir]
    else:
        projects = []
        if (input_dir / "pom.xml").exists():
            projects.append(input_dir)
        for p in input_dir.iterdir():
            if p.is_dir() and p.name != 'results' and (p / "pom.xml").exists():
                projects.append(p)

    successful = []
    total_Tce = 0
    results = {}
    
    for project_path in sorted(projects):
        success, commented = process_project(project_path, args.timeout)
        total_Tce += commented
        results[project_path.name] = {'success': success, 'Tce': commented}
        if success:
            successful.append(project_path.name)

    # Save compile results
    compile_results = {
        'total_Tce': total_Tce,
        'projects': results
    }
    results_file = input_dir / "compile_results.json"
    with open(results_file, 'w') as f:
        json.dump(compile_results, f, indent=2)
    
    print(f"\nSuccessful: {len(successful)}/{len(projects)}")
    print(f"Total Tce (commented assertions): {total_Tce}")
    print(f"Results saved to: {results_file}")
    for name in successful:
        print(f"  ✓ {name}")


if __name__ == '__main__':
    main()


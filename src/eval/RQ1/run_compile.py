import shutil
import subprocess
import argparse
import json
from pathlib import Path
from comment_incompatible_assertions import comment_assertions

def run_maven_compile(project_path: Path):
    # compile project with maven
    cmd = "mvn test-compile -fae -B -Drat.skip=true"
    # get result from shell subprocess
    result = subprocess.run(cmd, shell=True, cwd=str(project_path), capture_output=True, text=True)
    log = result.stdout + "\n" + result.stderr
    # write error log to file
    error_log = project_path / "compilation_error.txt"
    with open(error_log, 'w', encoding='utf-8') as f:
        f.write(log)
    
    return log

def process_project(project_path: Path):
    print(f"\nProcessing: {project_path.name}")
    total_commented = 0
    # compile until success to make when run test no compile error
    while True:
        log = run_maven_compile(project_path)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--projects', type=str, nargs='*', default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    
    # processing multiple modules in one projects
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
        success, commented = process_project(project_path)
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
        print(f"  Done: {name}")

if __name__ == "__main__":
    main()
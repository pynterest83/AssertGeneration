import shutil
import subprocess
import argparse
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


def process_project(project_path: Path, timeout: int = 300) -> bool:
    print(f"\nProcessing: {project_path.name}")
    
    while True:
        log = run_maven_compile(project_path, timeout)
        
        if 'BUILD SUCCESS' in log:
            print(f"  BUILD SUCCESS")
            return True
        
        if '.java:[' not in log:
            print(f"  BUILD FAILED (no compilation errors)")
            return False
        
        error_log = project_path / "compilation_error.txt"
        commented = comment_assertions(str(error_log))
        print(f"  Commented {commented} lines")
        
        if commented == 0:
            return False


def copy_project(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=lambda d, f: [x for x in f if x in ['target', '.git']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--projects', type=str, nargs='*', default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.projects:
        projects = [input_dir / p for p in args.projects if (input_dir / p).is_dir()]
    else:
        projects = [p for p in input_dir.iterdir() if p.is_dir() and p.name != 'results']

    successful = []
    for project_path in sorted(projects):
        if process_project(project_path, args.timeout):
            copy_project(project_path, output_dir / project_path.name)
            successful.append(project_path.name)

    print(f"\nSuccessful: {len(successful)}/{len(projects)}")
    for name in successful:
        print(f"  ✓ {name}")


if __name__ == '__main__':
    main()


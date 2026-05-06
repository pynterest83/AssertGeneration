import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path


def find_projects(data_dir):
    projects = []
    for name in sorted(os.listdir(data_dir)):
        project_dir = os.path.join(data_dir, name)
        if not os.path.isdir(project_dir):
            continue
        # Check for pit.sh at root or in any subdirectory
        if _find_modules(project_dir):
            projects.append(name)
    return projects


def _find_modules(project_dir):
    modules = []
    for root, dirs, files in os.walk(project_dir):
        # Skip hidden dirs and build output
        dirs[:] = [d for d in dirs if not d.startswith('.')
                   and d not in ('target', 'node_modules', '__pycache__')]
        if 'pit.sh' in files:
            rel = os.path.relpath(root, project_dir)
            if rel == '.':
                rel = ''
            modules.append((rel, root))
    # Sort so modules run in deterministic order
    modules.sort(key=lambda x: x[0])
    return modules


def run_module_pit(module_name, module_dir, timeout=3600):
    pit_sh = os.path.join(module_dir, 'pit.sh')
    log_file = os.path.join(module_dir, 'pit_run.log')

    label = module_name or '(root)'
    print(f"    [{label}] Running pit.sh (timeout={timeout}s)...")
    print(f"    [{label}] Log: {log_file}")

    start = time.time()
    try:
        with open(log_file, 'w') as log:
            proc = subprocess.Popen(
                ['bash', 'pit.sh'],
                cwd=module_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            try:
                proc.communicate(timeout=timeout)
                elapsed = time.time() - start
                if proc.returncode == 0:
                    print(f"    [{label}] SUCCESS ({elapsed:.1f}s)")
                    return True, elapsed
                else:
                    print(f"    [{label}] FAILED (exit code {proc.returncode}, {elapsed:.1f}s)")
                    return False, elapsed
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.communicate()
                elapsed = time.time() - start
                print(f"    [{label}] TIMEOUT after {elapsed:.1f}s")
                return False, elapsed
            except KeyboardInterrupt:
                print(f"\n    [{label}] [Ctrl+C] Killing process group...")
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.communicate()
                raise

    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = time.time() - start
        print(f"    [{label}] ERROR: {e}")
        return False, elapsed


def check_module_outputs(module_name, module_dir):
    """Check which mutations.xml and pit txt files were generated for a module."""
    label = module_name or '(root)'
    suites = {
        'es (src)': 'target/pit-reports',
        'llm_oracle': 'llm_oracle/target/pit-reports',
        'no_oracle': 'no_oracle/target/pit-reports',
    }
    found = {}
    for suite_name, rel_path in suites.items():
        pit_dir = os.path.join(module_dir, rel_path)
        mutations_xml = None
        if os.path.exists(pit_dir):
            for subdir in sorted(os.listdir(pit_dir), reverse=True):
                candidate = os.path.join(pit_dir, subdir, 'mutations.xml')
                if os.path.exists(candidate):
                    mutations_xml = candidate
                    break
            direct = os.path.join(pit_dir, 'mutations.xml')
            if not mutations_xml and os.path.exists(direct):
                mutations_xml = direct

        found[suite_name] = mutations_xml
        status = "✓" if mutations_xml else "✗"
        print(f"    [{label}] {status} {suite_name}: {mutations_xml or 'NOT FOUND'}")

    return found


def main():
    parser = argparse.ArgumentParser(
        description='Run PIT mutation testing for RQ2 projects (per-module)')
    parser.add_argument('--data_dir', required=True,
                        help='Path to data/RQ2/output')
    parser.add_argument('--projects', nargs='*', default=None,
                        help='Specific project names (default: all)')
    parser.add_argument('--modules', nargs='*', default=None,
                        help='Specific module names within a project (e.g. client extras/guava)')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per module in seconds (default: 3600)')
    parser.add_argument('--check_only', action='store_true',
                        help='Only check existing PIT outputs, do not run')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    if args.projects:
        projects = args.projects
    else:
        projects = find_projects(data_dir)

    print(f"Projects: {projects}")
    print()

    results = {}
    for project in projects:
        project_dir = os.path.join(data_dir, project)
        print("=" * 60)
        print(f"Project: {project}")
        print("=" * 60)

        if not os.path.isdir(project_dir):
            print(f"  ERROR: Directory not found: {project_dir}")
            results[project] = {'error': 'not found'}
            continue

        modules = _find_modules(project_dir)
        if not modules:
            print(f"  WARNING: No pit.sh found in any module")
            results[project] = {'error': 'no pit.sh'}
            continue

        # Filter modules if --modules specified
        if args.modules:
            modules = [(name, path) for name, path in modules
                       if name in args.modules]

        print(f"  Modules: {[m[0] or '(root)' for m in modules]}")

        project_results = {}
        for module_name, module_dir in modules:
            print()
            if args.check_only:
                found = check_module_outputs(module_name, module_dir)
                project_results[module_name] = {
                    'outputs': {k: v is not None for k, v in found.items()}
                }
            else:
                success, elapsed = run_module_pit(
                    module_name, module_dir, args.timeout)
                found = check_module_outputs(module_name, module_dir)
                project_results[module_name] = {
                    'success': success,
                    'elapsed_seconds': elapsed,
                    'outputs': {k: v is not None for k, v in found.items()}
                }

        results[project] = project_results
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for project, modules_info in results.items():
        print(f"  {project}:")
        if 'error' in modules_info:
            print(f"    ERROR: {modules_info['error']}")
            continue
        for module_name, info in modules_info.items():
            label = module_name or '(root)'
            if args.check_only:
                outputs = info.get('outputs', {})
                status = "ALL" if all(outputs.values()) else "PARTIAL"
                print(f"    {label}: {status}")
            else:
                status = "OK" if info.get('success') else "FAIL"
                elapsed = info.get('elapsed_seconds', 0)
                print(f"    {label}: {status} ({elapsed:.0f}s)")


if __name__ == '__main__':
    main()

import os
import re
import sys
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path


# Liệt kê các project con trong data_dir có ít nhất 1 module chứa pit.sh.
def find_projects(data_dir):
    return [
        name for name in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, name))
           and _find_modules(os.path.join(data_dir, name))
    ]


# Tìm các module Maven (folder có pit.sh) trong 1 project.
def _find_modules(project_dir):
    modules = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')
                   and d not in ('target', 'node_modules', '__pycache__')]
        if 'pit.sh' in files:
            rel = os.path.relpath(root, project_dir)
            modules.append(('' if rel == '.' else rel, root))
    modules.sort(key=lambda x: x[0])
    return modules


# Chạy subprocess với timeout + SIGTERM/SIGINT cho cả process group; trả (returncode, timed_out).
def _run_proc(cmd, cwd, log, timeout):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    try:
        proc.communicate(timeout=timeout)
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.communicate()
        return -1, True
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.communicate()
        raise


# Chạy full pit.sh cho 1 module (mvn clean test x3 + pitest x3), log ra pit_run.log.
def run_module_pit(module_name, module_dir, timeout=None):
    label = module_name or '(root)'
    log_file = os.path.join(module_dir, 'pit_run.log')
    print(f"  [{label}] pit.sh ...")
    start = time.time()
    with open(log_file, 'w') as log:
        rc, timed_out = _run_proc(['bash', 'pit.sh'], module_dir, log, timeout)
    elapsed = time.time() - start
    if timed_out:
        print(f"  [{label}] TIMEOUT ({elapsed:.1f}s)")
        return False, elapsed
    ok = rc == 0
    print(f"  [{label}] {'OK' if ok else f'FAILED exit={rc}'} ({elapsed:.1f}s)")
    return ok, elapsed


SUITES = ['src', 'llm_oracle', 'no_oracle']


# Build path file *_ESTest.java từ tên class FQN trong 1 suite cụ thể.
def _find_test_file(module_dir, class_fqn, suite):
    return os.path.join(module_dir, suite, 'test', 'java', class_fqn.replace('.', '/') + '.java')


# Parse log PIT, trả set class báo "did not pass without mutation" (DNP).
def _parse_dnp_classes(log_file):
    dnp_re = re.compile(r'SEVERE.*testClass=([\w.]+),\s*name=\1\].*did not pass without mutation')
    classes = set()
    if not os.path.exists(log_file):
        return classes
    with open(log_file, 'r', errors='ignore') as f:
        for line in f:
            m = dnp_re.search(line)
            if m:
                classes.add(m.group(1))
    return classes


# Comment toàn bộ @Test trong các class DNP, áp đồng bộ cho cả 3 suite.
def _comment_dnp_classes(module_dir, dnp_classes, log_fh=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_for_pit import comment_tests_in_file, get_test_methods
    total = 0
    for cls in sorted(dnp_classes):
        for suite in SUITES:
            fp = _find_test_file(module_dir, cls, suite)
            if os.path.exists(fp):
                total += comment_tests_in_file(fp, get_test_methods(fp))
    if log_fh:
        log_fh.write(f"[DNP] commented {total} @Test across {len(dnp_classes)} class(es)\n")
    return total


# Recompile cả 3 suite sau khi đã comment DNP (chạy "mvn test-compile" thay cho "mvn clean test").
def _recompile_all_suites(module_dir, log_fh=None):
    with open(os.path.join(module_dir, 'pit.sh'), 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('#') or not s:
                continue
            if 'mvn clean test' in s and 'pitest' not in s:
                compile_cmd = s.replace('mvn clean test', 'mvn test-compile', 1)
                subprocess.run(['bash', '-c', compile_cmd], cwd=module_dir,
                               stdout=log_fh, stderr=subprocess.STDOUT)


# Parse pit.sh, chỉ lấy các dòng "mvn pitest:mutationCoverage" (bỏ phần mvn clean test).
def parse_pit_only_commands(module_dir):
    commands = []
    with open(os.path.join(module_dir, 'pit.sh'), 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if 'pitest-maven:mutationCoverage' in s:
                commands.append(s)
    return commands


# Chạy chỉ phần pitest cho 1 module với cơ chế tự retry khi gặp DNP, có thể lọc theo --suites.
def run_module_pit_only(module_name, module_dir, timeout=None, suites=None):
    label = module_name or '(root)'
    log_file = os.path.join(module_dir, 'pit_run.log')

    commands = parse_pit_only_commands(module_dir)
    if suites:
        commands = [c for c in commands if any(f'-Dtest.dir={s}' in c for s in suites)]
    if not commands:
        print(f"  [{label}] WARNING: No pitest commands found")
        return False, 0.0

    overall_start = time.time()
    all_ok = True

    dnp_file = os.path.join(module_dir, 'pit_dnp.json')
    module_dnp = set()
    if os.path.exists(dnp_file):
        with open(dnp_file) as f:
            module_dnp = set(json.load(f))

    with open(log_file, 'a' if suites else 'w') as log:
        if suites and module_dnp:
            _comment_dnp_classes(module_dir, module_dnp, log)
            _recompile_all_suites(module_dir, log)

        completed = []
        needs_rerun = set()

        for i, cmd in enumerate(commands, 1):
            m_suite = re.search(r'-Dtest\.dir=(\S+)', cmd)
            suite_name = m_suite.group(1) if m_suite else f'suite{i}'

            cmd_ok = False
            for attempt in range(1, 100):
                attempt_label = f" retry={attempt}" if attempt > 1 else ""
                print(f"  [{label}] [{i}/{len(commands)}] {suite_name}{attempt_label} ...")
                log.write(f"\n{'='*60}\n[{i}/{len(commands)}] attempt={attempt} {cmd}\n{'='*60}\n")
                log.flush()

                start = time.time()
                rc, timed_out = _run_proc(['bash', '-c', cmd], module_dir, log, timeout)
                elapsed = time.time() - start

                if timed_out:
                    print(f"  [{label}] [{i}/{len(commands)}] TIMEOUT ({elapsed:.1f}s)")
                    break

                if rc == 0:
                    print(f"  [{label}] [{i}/{len(commands)}] {suite_name}: OK ({elapsed:.1f}s)")
                    completed.append((i, cmd, suite_name))
                    cmd_ok = True
                    break

                log.flush()
                new_dnp = _parse_dnp_classes(log_file) - module_dnp
                if new_dnp:
                    module_dnp.update(new_dnp)
                    print(f"  [{label}] [{i}/{len(commands)}] {suite_name}: exit={rc}, {len(new_dnp)} DNP — retrying: {sorted(new_dnp)}")
                    _comment_dnp_classes(module_dir, new_dnp, log)
                    _recompile_all_suites(module_dir, log)
                    needs_rerun.update(pi for pi, *_ in completed)
                    continue

                print(f"  [{label}] [{i}/{len(commands)}] {suite_name}: exit={rc} ({elapsed:.1f}s)")
                break

            if not cmd_ok:
                all_ok = False

    if needs_rerun:
        with open(log_file, 'a') as log:
            for pi, pcmd, psuite in (t for t in completed if t[0] in needs_rerun):
                log.write(f"\n{'='*60}\n[{pi}] re-run: {pcmd}\n{'='*60}\n")
                log.flush()
                rc, _ = _run_proc(['bash', '-c', pcmd], module_dir, log, timeout)
                print(f"  [{label}] [{pi}/?] {psuite} re-run: {'OK' if rc == 0 else f'exit={rc}'}")
                if rc != 0:
                    all_ok = False

    if module_dnp:
        with open(dnp_file, 'w') as f:
            json.dump(sorted(module_dnp), f, indent=2)

    elapsed = time.time() - overall_start
    print(f"  [{label}] {'SUCCESS' if all_ok else 'FAILED'} ({elapsed:.1f}s)")
    return all_ok, elapsed


# Kiểm tra 3 suite đã có file mutations.xml hay chưa (1 dict per-suite -> path hoặc None).
def check_module_outputs(module_name, module_dir):
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
    return found


# Entry point: lặp các project, chạy PIT cho từng module, in summary OK/FAIL.
def main():
    parser = argparse.ArgumentParser(description='Run PIT mutation testing for RQ2 projects')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--projects', nargs='*', default=None)
    parser.add_argument('--modules', nargs='*', default=None)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--check_only', action='store_true')
    parser.add_argument('--pit_only', action='store_true',
                        help='Skip mvn clean test, only run mvn pitest commands')
    parser.add_argument('--suites', nargs='*', default=None,
                        help='Only run specific suites (implies --pit_only)')
    args = parser.parse_args()
    if args.suites:
        args.pit_only = True

    data_dir = os.path.abspath(args.data_dir)
    projects = args.projects or find_projects(data_dir)
    print(f"Projects: {projects}\n")

    results = {}
    for project in projects:
        project_dir = os.path.join(data_dir, project)
        print("=" * 60)
        print(f"Project: {project}")
        print("=" * 60)

        if not os.path.isdir(project_dir):
            results[project] = {'error': 'not found'}
            continue

        modules = _find_modules(project_dir)
        if not modules:
            results[project] = {'error': 'no pit.sh'}
            continue

        if args.modules:
            modules = [(name, path) for name, path in modules if name in args.modules]

        print(f"  Modules: {[m[0] or '(root)' for m in modules]}")

        project_results = {}
        for module_name, module_dir in modules:
            print()
            found = None
            if args.check_only:
                found = check_module_outputs(module_name, module_dir)
                project_results[module_name] = {'outputs': {k: v is not None for k, v in found.items()}}
                continue

            if args.pit_only:
                success, elapsed = run_module_pit_only(module_name, module_dir, args.timeout, args.suites)
            else:
                success, elapsed = run_module_pit(module_name, module_dir, args.timeout)

            found = check_module_outputs(module_name, module_dir)
            project_results[module_name] = {
                'success': success,
                'elapsed_seconds': elapsed,
                'outputs': {k: v is not None for k, v in found.items()}
            }

        results[project] = project_results
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for project, modules_info in results.items():
        print(f"  {project}:")
        if 'error' in modules_info:
            print(f"    {modules_info['error']}")
            continue
        for module_name, info in modules_info.items():
            label = module_name or '(root)'
            if args.check_only:
                outputs = info.get('outputs', {})
                print(f"    {label}: {'ALL' if all(outputs.values()) else 'PARTIAL'}")
            else:
                print(f"    {label}: {'OK' if info.get('success') else 'FAIL'} ({info.get('elapsed_seconds', 0):.0f}s)")


if __name__ == '__main__':
    main()

import os
import re
import json
import time
import signal
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SUITES = ['src', 'llm_oracle', 'no_oracle']

# Chuẩn hóa tên test "test001" -> "test1" để match giữa các suite (ES vs LLM).
def normalize_test_name(name):
    m = re.match(r'test0*(\d+)$', name)
    return f"test{m.group(1)}" if m else name

# Tìm Tce — test bị compile error ở RQ1 (line đã chèn //COMPILE_ERROR), trả về dict file->set(tên test).
def find_tce_tests(project_dir):
    method_re = re.compile(r'public\s+void\s+(test\d+)\s*\(')
    file_to_excluded = defaultdict(set)

    for root, dirs, files in os.walk(project_dir):
        if 'llm_oracle' not in Path(root).parts:
            continue
        for name in files:
            if not name.endswith('_ESTest.java'):
                continue
            filepath = os.path.join(root, name)
            current_test = None
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    m = method_re.search(line)
                    if m:
                        current_test = m.group(1)
                    if '//COMPILE_ERROR' in line and current_test:
                        file_to_excluded[filepath].add(current_test)

    return file_to_excluded

# Tìm Tfp — test fail/error khi chạy ở RQ1 (parse surefire-reports), trả dict file->set(tên test).
def find_tfp_tests(surefire_bases, project_dir):
    failed_re = re.compile(r'(\w+)\(([\w.]+)\)\s+Time elapsed:.*<<<\s+(FAILURE|ERROR)!')
    class_to_excluded = defaultdict(set)

    for surefire_base in surefire_bases:
        for root, dirs, files in os.walk(surefire_base):
            if 'surefire-reports' not in root:
                continue
            for name in files:
                if not name.endswith('.txt') or name.endswith('-output.txt'):
                    continue
                with open(os.path.join(root, name), 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        match = failed_re.search(line)
                        if match:
                            class_to_excluded[match.group(2)].add(match.group(1))

    if not class_to_excluded:
        return defaultdict(set)

    llm_files = {}
    for root, dirs, files in os.walk(project_dir):
        if 'llm_oracle' not in Path(root).parts:
            continue
        for name in files:
            if name.endswith('_ESTest.java'):
                llm_files.setdefault(name, []).append(os.path.join(root, name))

    file_to_excluded = defaultdict(set)
    for test_class, tests in class_to_excluded.items():
        simple_name = test_class.split('.')[-1] + '.java'
        relative = test_class.replace('.', os.sep) + '.java'
        candidates = llm_files.get(simple_name, [])
        for cpath in candidates:
            if relative.replace(os.sep, '/') in cpath.replace(os.sep, '/'):
                file_to_excluded[cpath].update(tests)
                break
        else:
            if len(candidates) == 1:
                file_to_excluded[candidates[0]].update(tests)

    return file_to_excluded


# Lấy danh sách test methods trong 1 file Java; normalize=True thì map về dạng "testN" để so sánh giữa suite.
def get_test_methods(filepath, normalize=False):
    methods = {} if normalize else set()
    if not os.path.exists(filepath):
        return methods

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'public\s+void\s+(test\w+)\s*\(', line)
            if m:
                original = m.group(1)
                if normalize:
                    num_match = re.match(r't[eE]st0*(\d+)$', original)
                    if num_match:
                        methods["test" + num_match.group(1)] = original
                    else:
                        methods[original] = original
                else:
                    methods.add(original)
    return methods

# Tìm test exception (có trong src nhưng không có trong llm_oracle) + test "thừa" trong llm_oracle để loại khỏi cả 3 suite.
def find_exception_tests(project_dir):
    llm_files = {}
    src_files = {}

    for root, dirs, files in os.walk(project_dir):
        for name in files:
            if not name.endswith('_ESTest.java'):
                continue
            filepath = os.path.join(root, name)
            rel_parts = list(Path(os.path.relpath(filepath, project_dir)).parts)
            for i, p in enumerate(rel_parts):
                if p in ('src', 'llm_oracle', 'no_oracle'):
                    key = os.path.join(*rel_parts[:i], *rel_parts[i+1:])
                    if p == 'llm_oracle':
                        llm_files[key] = filepath
                    elif p == 'src':
                        src_files[key] = filepath
                    break

    src_excluded = defaultdict(set)
    llm_extra_excluded = defaultdict(set)

    for key, src_path in src_files.items():
        llm_path = llm_files.get(key)
        if not llm_path:
            src_methods = get_test_methods(src_path)
            if src_methods:
                src_excluded[src_path] = src_methods
            continue

        src_methods = get_test_methods(src_path, normalize=False)
        llm_methods_map = get_test_methods(llm_path, normalize=True)
        llm_normalized = set(llm_methods_map.keys())

        missing_from_llm = src_methods - llm_normalized
        if missing_from_llm:
            src_excluded[src_path] = missing_from_llm

        extra_in_llm = llm_normalized - src_methods
        if extra_in_llm:
            llm_extra_excluded[llm_path] = {llm_methods_map[n] for n in extra_in_llm}

    return src_excluded, llm_extra_excluded


# Comment // các @Test có tên trong excluded_method_names, tắt separateClassLoader, chèn test_nothing dummy nếu cần.
def comment_tests_in_file(filepath, excluded_method_names, add_dummy=True):
    if not os.path.exists(filepath):
        return 0

    method_re = re.compile(r'public\s+void\s+(test\d+)\s*\(')
    normalized_excluded = {normalize_test_name(n) for n in excluded_method_names}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if 'void test_nothing(' in "".join(lines):
        add_dummy = False

    commented = 0
    new_lines = []
    i = 0
    dummy_added = False
    while i < len(lines):
        line = lines[i]

        if 'separateClassLoader = true' in line:
            line = line.replace('separateClassLoader = true', 'separateClassLoader = false')

        if add_dummy and not dummy_added and 'public class' in line and '_ESTest ' in line:
            new_lines.append(line)
            new_lines.append('  @Test(timeout = 4000)\n')
            new_lines.append('  public void test_nothing()  throws Throwable  {}\n')
            dummy_added = True
            i += 1
            continue

        if '@Test' in line:
            method_name = None
            for j in range(i, min(i + 5, len(lines))):
                m = method_re.search(lines[j])
                if m:
                    method_name = m.group(1)
                    break
            if method_name and normalize_test_name(method_name) in normalized_excluded:
                line = '//' + line
                commented += 1

        new_lines.append(line)
        i += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    return commented


# Từ path file trong llm_oracle, suy ra path tương ứng trong src và no_oracle (để comment đồng bộ 3 suite).
def get_corresponding_paths(llm_oracle_file, project_dir):
    parts = list(Path(os.path.relpath(llm_oracle_file, project_dir)).parts)
    for i, part in enumerate(parts):
        if part == 'llm_oracle':
            src_file = os.path.join(project_dir, *parts[:i], 'src', *parts[i+1:])
            no_oracle_file = os.path.join(project_dir, *parts[:i], 'no_oracle', *parts[i+1:])
            return src_file, no_oracle_file
    return None, None

# Tắt separateClassLoader cho mọi *_ESTest.java chưa được xử lý (PIT yêu cầu cùng classloader).
def disable_classloader_all(project_dir, already_processed):
    for root, dirs, files in os.walk(project_dir):
        if '.evosuite' in Path(root).parts:
            continue
        for name in files:
            if not name.endswith('_ESTest.java'):
                continue
            filepath = os.path.join(root, name)
            if filepath in already_processed:
                continue

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            dummy_exists = 'void test_nothing(' in "".join(lines)
            new_lines = []
            modified = False
            dummy_added = False
            for line in lines:
                if 'separateClassLoader = true' in line:
                    line = line.replace('separateClassLoader = true', 'separateClassLoader = false')
                    modified = True
                if not dummy_exists and not dummy_added and 'public class' in line and '_ESTest ' in line:
                    new_lines.append(line)
                    new_lines.append('  @Test(timeout = 4000)\n')
                    new_lines.append('  public void test_nothing()  throws Throwable  {}\n')
                    dummy_added = True
                    modified = True
                    continue
                new_lines.append(line)

            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)


# Đếm số @Test còn active (chưa bị comment) trong 1 file.
def count_active_tests(filepath):
    if not os.path.exists(filepath):
        return -1
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.lstrip().startswith('@Test'):
                count += 1
    return count


# Kiểm tra 3 suite có cùng số @Test active (fairness cho mutation testing), trả (ok, mismatches, totals).
def verify_test_counts(project_dir):
    file_map = {}
    for root, dirs, files in os.walk(project_dir):
        if '.evosuite' in Path(root).parts:
            continue
        for name in files:
            if not name.endswith('_ESTest.java'):
                continue
            filepath = os.path.join(root, name)
            rel_parts = list(Path(os.path.relpath(filepath, project_dir)).parts)
            suite_name = None
            for i, p in enumerate(rel_parts):
                if p in ('src', 'llm_oracle', 'no_oracle'):
                    suite_name = p
                    key = os.path.join(*rel_parts[:i], *rel_parts[i+1:])
                    break
            if not suite_name:
                continue
            file_map.setdefault(key, {})[suite_name] = (filepath, count_active_tests(filepath))

    mismatches = []
    suite_totals = defaultdict(int)
    for key, suites in sorted(file_map.items()):
        counts = {s: info[1] for s, info in suites.items()}
        for s, cnt in counts.items():
            suite_totals[s] += cnt
        if len(suites) >= 2 and len(set(counts.values())) > 1:
            mismatches.append({'file': key, 'counts': counts})

    return len(mismatches) == 0, mismatches, dict(suite_totals)


# Trích chữ số trong tên test ("test42" -> 42) để sort theo số.
def get_test_num(name):
    m = re.search(r'\d+', name)
    return int(m.group()) if m else -1


# Copy nguyên cây thư mục input -> output, bỏ qua các folder target/.
def copy_input_to_output(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    def ignore_targets(src, names):
        return {n for n in names if n == 'target' and os.path.isdir(os.path.join(src, n))}

    shutil.copytree(input_dir, output_dir, ignore=ignore_targets)


# Parse pit.sh, lấy ra các dòng "mvn clean test" (cho từng suite) kèm test.dir và target.dir.
def _parse_pit_sh_clean_test(module_dir):
    test_dir_re = re.compile(r'-Dtest\.dir=(\S+)')
    target_dir_re = re.compile(r'-Dtarget\.dir=(\S+)')
    suites = []
    with open(os.path.join(module_dir, 'pit.sh'), 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('#') or not s:
                continue
            if 'mvn clean test' in s and 'pitest' not in s:
                m_test = test_dir_re.search(s)
                m_target = target_dir_re.search(s)
                if m_test and m_target:
                    suites.append((m_test.group(1).rstrip('/'), m_target.group(1).rstrip('/'), s))
    return suites


# Chạy "mvn clean test <args>" trong subprocess, trả (returncode, elapsed); -1 nếu timeout.
def _run_mvn_clean_test(module_dir, args_str, log_file, timeout=600):
    start = time.time()
    with open(log_file, 'a') as log:
        proc = subprocess.Popen(['bash', '-c', f"mvn clean test {args_str}"], cwd=module_dir,
                                stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        try:
            proc.communicate(timeout=timeout)
            return proc.returncode, time.time() - start
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.communicate()
            return -1, time.time() - start
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.communicate()
            raise


# Parse surefire-reports/*.txt, trả (per_test fail/error theo class, set class bị init error).
def _parse_surefire_reports(surefire_dir):
    per_test_re = re.compile(r'(\w+)\(([\w.]+)\)\s+Time elapsed:.*<<<\s+(FAILURE|ERROR)!')
    class_err_re = re.compile(r'^([\w.]+_ESTest)\s+Time elapsed:.*<<<\s+ERROR!')
    per_test = defaultdict(set)
    class_errs = set()
    if not os.path.isdir(surefire_dir):
        return per_test, class_errs
    for fname in os.listdir(surefire_dir):
        if not fname.endswith('.txt') or fname.endswith('-output.txt'):
            continue
        with open(os.path.join(surefire_dir, fname), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = per_test_re.search(line)
                if m:
                    method, cls = m.group(1), m.group(2)
                    if method == 'initializationError':
                        class_errs.add(cls)
                    else:
                        per_test[cls].add(method)
                    continue
                m2 = class_err_re.match(line.strip())
                if m2:
                    class_errs.add(m2.group(1))
    return per_test, class_errs


# Parse "Crashed tests:" block trong log Maven, trả set tên class _ESTest đã crash.
def _parse_crashed_tests(log_file, offset=0):
    crashed = set()
    crashed_re = re.compile(r'\[ERROR\]\s+([\w.]+_ESTest)\s*$')
    in_block = False
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(offset)
        for line in f:
            if 'Crashed tests:' in line:
                in_block = True
                continue
            if in_block:
                m = crashed_re.match(line.strip())
                if m:
                    crashed.add(m.group(1))
                elif line.strip() and not line.strip().startswith('[ERROR]'):
                    in_block = False
    return crashed

# Build và lọc tiếp: chạy mvn clean test cho từng suite, comment các test fail/crash để cuối cùng cả 3 suite đều "green".
def make_green_suite(module_dir, timeout=600):
    label = os.path.basename(module_dir) or '(root)'
    suite_configs = _parse_pit_sh_clean_test(module_dir)
    if not suite_configs:
        return False

    log_file = os.path.join(module_dir, 'green_suite_mvn.log')
    open(log_file, 'w').close()

    all_per_test = defaultdict(set)
    all_class_errs = set()

    for suite_name, target_dir, original_line in suite_configs:
        args_str = original_line[len('mvn clean test'):].strip()
        log_offset = os.path.getsize(log_file)
        rc, elapsed = _run_mvn_clean_test(module_dir, args_str, log_file, timeout)
        print(f"  [{label}] [{suite_name}] {'OK' if rc == 0 else f'exit={rc}'} ({elapsed:.1f}s)")

        per_test, class_errs = _parse_surefire_reports(
            os.path.join(module_dir, target_dir, 'surefire-reports'))
        class_errs.update(_parse_crashed_tests(log_file, log_offset))

        for cls, methods in per_test.items():
            all_per_test[cls].update(methods)
        all_class_errs.update(class_errs)

    for cls in all_class_errs:
        for suite in SUITES:
            fp = os.path.join(module_dir, suite, 'test', 'java', cls.replace('.', '/') + '.java')
            if os.path.exists(fp):
                all_per_test[cls].update(get_test_methods(fp))
                break

    total_commented = 0
    for cls in sorted(all_per_test):
        methods = all_per_test[cls]
        if not methods:
            continue
        for suite in SUITES:
            fp = os.path.join(module_dir, suite, 'test', 'java', cls.replace('.', '/') + '.java')
            if os.path.exists(fp):
                total_commented += comment_tests_in_file(fp, methods)

    print(f"  [{label}] Total commented: {total_commented}")
    results = {
        'per_test_commented': {cls: sorted(m) for cls, m in all_per_test.items() if m},
        'class_errors': list(all_class_errs),
        'suites_run': [s[0] for s in suite_configs],
        'total_commented': total_commented,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(module_dir, 'green_suite_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return True


# Liệt kê các module Maven (folder có pit.sh) trong project_dir.
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


# Entry point: gom Tce/Tfp/exception, comment đồng bộ 3 suite, verify, rồi make_green_suite từng module.
def main():
    parser = argparse.ArgumentParser(description='Prepare test suites for PIT mutation testing')
    parser.add_argument('--input_dir', default=None)
    parser.add_argument('--project_dir', required=True)
    parser.add_argument('--surefire_base', required=True, nargs='+')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    surefire_bases = [os.path.abspath(p) for p in args.surefire_base]

    if args.input_dir and not args.dry_run:
        copy_input_to_output(os.path.abspath(args.input_dir), project_dir)

    tce_excluded = find_tce_tests(project_dir)
    tce_count = sum(len(v) for v in tce_excluded.values())

    tfp_excluded = find_tfp_tests(surefire_bases, project_dir)
    tfp_count = sum(len(v) for v in tfp_excluded.values())

    exc_excluded, llm_extra_excluded = find_exception_tests(project_dir)
    exc_count = sum(len(v) for v in exc_excluded.values())
    llm_extra_count = sum(len(v) for v in llm_extra_excluded.values())

    all_excluded_llm = defaultdict(set)
    for f, names in tce_excluded.items():
        all_excluded_llm[f].update(names)
    for f, names in tfp_excluded.items():
        all_excluded_llm[f].update(names)

    total_excluded = sum(len(v) for v in all_excluded_llm.values()) + exc_count + llm_extra_count

    if args.dry_run:
        ok, mismatches, suite_totals = verify_test_counts(project_dir)
        for s in ('src', 'llm_oracle', 'no_oracle'):
            print(f"  {s}: {suite_totals.get(s, 0)} active @Test")
        if not ok:
            for mm in mismatches:
                print(f"  {mm['file']}: {mm['counts']}")
        return

    llm_exclusions = defaultdict(set)
    src_exclusions = defaultdict(set)
    no_exclusions = defaultdict(set)

    for llm_file, methods in all_excluded_llm.items():
        llm_exclusions[llm_file].update(methods)
        src_file, no_oracle_file = get_corresponding_paths(llm_file, project_dir)
        if src_file:
            src_exclusions[src_file].update(methods)
        if no_oracle_file:
            no_exclusions[no_oracle_file].update(methods)

    for src_file, methods in exc_excluded.items():
        src_exclusions[src_file].update(methods)
        parts = list(Path(os.path.relpath(src_file, project_dir)).parts)
        for i, p in enumerate(parts):
            if p == 'src':
                no_exclusions[os.path.join(project_dir, *parts[:i], 'no_oracle', *parts[i+1:])].update(methods)
                llm_exclusions[os.path.join(project_dir, *parts[:i], 'llm_oracle', *parts[i+1:])].update(methods)
                break

    for llm_file, methods in llm_extra_excluded.items():
        llm_exclusions[llm_file].update(methods)

    total_commented = {'llm_oracle': 0, 'src': 0, 'no_oracle': 0}
    all_processed_files = set()

    for llm_file, methods in sorted(llm_exclusions.items()):
        total_commented['llm_oracle'] += comment_tests_in_file(llm_file, methods)
        all_processed_files.add(llm_file)

    for src_file, methods in sorted(src_exclusions.items()):
        total_commented['src'] += comment_tests_in_file(src_file, methods)
        all_processed_files.add(src_file)

    for no_file, methods in sorted(no_exclusions.items()):
        total_commented['no_oracle'] += comment_tests_in_file(no_file, methods)
        all_processed_files.add(no_file)

    disable_classloader_all(project_dir, all_processed_files)

    ok, mismatches, suite_totals = verify_test_counts(project_dir)
    if not ok:
        for mm in mismatches:
            print(f"MISMATCH {mm['file']}: {mm['counts']}")

    results = {
        'Tce_tests': tce_count,
        'Tfp_tests': tfp_count,
        'exception_tests': exc_count,
        'llm_extra_tests': llm_extra_count,
        'total_excluded': total_excluded,
        'commented': total_commented,
        'tce_tfp_files': {
            os.path.relpath(f, project_dir): sorted(list(names), key=get_test_num)
            for f, names in all_excluded_llm.items()
        },
        'exception_files': {
            os.path.relpath(f, project_dir): sorted(list(names), key=get_test_num)
            for f, names in exc_excluded.items()
        },
        'llm_extra_files': {
            os.path.relpath(f, project_dir): sorted(list(names), key=get_test_num)
            for f, names in llm_extra_excluded.items()
        },
        'verification_passed': ok,
        'active_remaining': suite_totals,
        'verification_mismatches': mismatches,
    }
    results_file = os.path.join(project_dir, 'prepare_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Tce={tce_count} Tfp={tfp_count} exc={exc_count} total_excluded={total_excluded}")
    print(f"active: src={suite_totals.get('src',0)} llm={suite_totals.get('llm_oracle',0)} no={suite_totals.get('no_oracle',0)}")
    print(f"verify={'PASSED' if ok else f'FAILED({len(mismatches)})'} | {results_file}")

    for _, module_dir in _find_modules(project_dir):
        make_green_suite(module_dir, args.timeout)


if __name__ == '__main__':
    main()

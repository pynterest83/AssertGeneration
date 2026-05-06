"""prepare_for_pit.py

Prepares test suites for PIT mutation testing by:
1. (Optional) Copying project from --input_dir to --project_dir (output location)
2. Identifying Tce tests (compile errors → //COMPILE_ERROR) in llm_oracle
3. Identifying Tfp tests (false positives) from surefire reports
4. Commenting out @Test for those tests in ALL 3 suites (src, llm_oracle, no_oracle)
5. Adding test_nothing() dummy test to affected files
6. Disabling separateClassLoader for EvoSuite in ALL test files

Usage (with copy from input to output):
    python eval/RQ2/prepare_for_pit.py \
        --input_dir  data/RQ2/input/async-http-client \
        --project_dir data/RQ2/output/async-http-client \
        --surefire_base data/RQ1/output/async-http-client/injected_assertion

Usage (operate in-place, no copy):
    python eval/RQ2/prepare_for_pit.py \
        --project_dir data/RQ2/output/async-http-client \
        --surefire_base data/RQ1/output/async-http-client/injected_assertion

NOTE: llm_oracle may have fewer tests than src/no_oracle (exception tests skipped).
      This script matches by test METHOD NAME (e.g., "test30"), not by @Test index.
"""

import os
import re
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def normalize_test_name(name):
    """Normalize test method name: test05 -> test5, test00 -> test0."""
    m = re.match(r'test0*(\d+)$', name)
    return f"test{m.group(1)}" if m else name


# ---------------------------------------------------------------------------
# Step 1: Find Tce tests (//COMPILE_ERROR in llm_oracle)
# ---------------------------------------------------------------------------

def find_tce_tests(project_dir):
    """
    Scan llm_oracle test files for //COMPILE_ERROR lines.
    Returns: dict mapping llm_oracle_file_path -> set of test method names
             e.g. {"/.../ThreadSafeCookieStore_ESTest.java": {"test11", "test30"}}
    """
    method_re = re.compile(r'public\s+void\s+(test\d+)\s*\(')
    file_to_excluded = defaultdict(set)

    for root, dirs, files in os.walk(project_dir):
        parts = Path(root).parts
        if 'llm_oracle' not in parts:
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


# ---------------------------------------------------------------------------
# Step 2: Find Tfp tests (failures/errors from surefire reports)
# ---------------------------------------------------------------------------

def find_tfp_tests(surefire_bases, project_dir):
    """
    Parse surefire reports to find failing/error tests.
    Returns: dict mapping llm_oracle_file_path -> set of test method names
    """
    failed_re = re.compile(
        r'(\w+)\(([\w.]+)\)\s+Time elapsed:.*<<<\s+(FAILURE|ERROR)!'
    )
    # Collect: class_name -> set of test names
    class_to_excluded = defaultdict(set)

    for surefire_base in surefire_bases:
        for root, dirs, files in os.walk(surefire_base):
            if 'surefire-reports' not in root:
                continue
            for name in files:
                if not name.endswith('.txt') or name.endswith('-output.txt'):
                    continue
                filepath = os.path.join(root, name)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        match = failed_re.search(line)
                        if match:
                            test_name = match.group(1)   # e.g., "test30"
                            test_class = match.group(2)   # e.g., "org...ThreadSafeCookieStore_ESTest"
                            class_to_excluded[test_class].add(test_name)

    if not class_to_excluded:
        return defaultdict(set)
    llm_files = {}
    for root, dirs, files in os.walk(project_dir):
        parts = Path(root).parts
        if 'llm_oracle' not in parts:
            continue
        for name in files:
            if name.endswith('_ESTest.java'):
                llm_files.setdefault(name, []).append(os.path.join(root, name))

    file_to_excluded = defaultdict(set)
    for test_class, tests in class_to_excluded.items():
        # org.foo.Bar_ESTest -> Bar_ESTest.java
        simple_name = test_class.split('.')[-1] + '.java'
        # org.foo.Bar_ESTest -> org/foo/Bar_ESTest.java (for disambiguation)
        relative = test_class.replace('.', os.sep) + '.java'

        candidates = llm_files.get(simple_name, [])
        for cpath in candidates:
            normalized = cpath.replace(os.sep, '/')
            if relative.replace(os.sep, '/') in normalized:
                file_to_excluded[cpath].update(tests)
                break
        else:
            # Fallback: take first match if only one candidate
            if len(candidates) == 1:
                file_to_excluded[candidates[0]].update(tests)
            elif candidates:
                print(f"  WARNING: Ambiguous match for {test_class}, "
                      f"candidates: {candidates}")

    return file_to_excluded


# ---------------------------------------------------------------------------
# Step 3: Find exception tests (in src but NOT in llm_oracle)
# ---------------------------------------------------------------------------

def get_test_methods(filepath, normalize=False):
    """
    Extract all test method names from a Java test file.
    If normalize=True, converts 'test05' -> 'test5', 'test00' -> 'test0' 
    to handle zero-padding from RQ1 injection. Return a dict of normalized_name -> original_name.
    If normalize=False, returns a set of original names.
    """
    if normalize:
        methods = {}
    else:
        methods = set()
        
    if not os.path.exists(filepath):
        return methods
        
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'public\s+void\s+(test\w+)\s*\(', line)
            if m:
                original = m.group(1)
                if normalize:
                    # Strip leading zeros after 'test', e.g., 'test00' -> 'test0', 'test05' -> 'test5'
                    # Regex logic: find 'test' followed by zeros, replace with 'test', 
                    # except if it's all zeros, leave one zero ('test0').
                    num_match = re.match(r't[eE]st0*(\d+)$', original)
                    if num_match:
                        normalized = "test" + num_match.group(1)
                        methods[normalized] = original
                    else:
                        methods[original] = original
                else:
                    methods.add(original)
    return methods


def find_exception_tests(project_dir):
    """
    Find bidirectional method mismatches between src and llm_oracle:
    - Methods in src but NOT in llm_oracle (exception tests skipped during injection)
    - Methods in llm_oracle but NOT in src (extra methods from RQ1 output)
    Both must be excluded so all 3 suites run the exact same test methods.

    Returns: (src_excluded, llm_extra_excluded)
        src_excluded: dict mapping src_file_path -> set of method names
        llm_extra_excluded: dict mapping llm_oracle_file_path -> set of method names
    """
    # Build mapping: (module_prefix, filename) -> llm_oracle_path
    llm_files = {}  # relative_key -> full_path
    src_files = {}  # relative_key -> full_path

    for root, dirs, files in os.walk(project_dir):
        parts = Path(root).parts
        for name in files:
            if not name.endswith('_ESTest.java'):
                continue
            filepath = os.path.join(root, name)
            rel = os.path.relpath(filepath, project_dir)
            rel_parts = list(Path(rel).parts)

            # Find the suite dir (src/llm_oracle/no_oracle) and build a key
            for i, p in enumerate(rel_parts):
                if p in ('src', 'llm_oracle', 'no_oracle'):
                    # key = everything except the suite dir name
                    key = os.path.join(*rel_parts[:i], *rel_parts[i+1:])
                    if p == 'llm_oracle':
                        llm_files[key] = filepath
                    elif p == 'src':
                        src_files[key] = filepath
                    break

    src_excluded = defaultdict(set)      # methods in src but not llm
    llm_extra_excluded = defaultdict(set) # methods in llm but not src

    for key, src_path in src_files.items():
        llm_path = llm_files.get(key)
        if not llm_path:
            # Entire file missing from llm_oracle — all src tests are exception
            src_methods = get_test_methods(src_path)
            if src_methods:
                src_excluded[src_path] = src_methods
            continue

        src_methods = get_test_methods(src_path, normalize=False)
        llm_methods_map = get_test_methods(llm_path, normalize=True)
        llm_normalized = set(llm_methods_map.keys())

        # Methods in src but not llm_oracle
        missing_from_llm = src_methods - llm_normalized
        if missing_from_llm:
            src_excluded[src_path] = missing_from_llm

        # Methods in llm_oracle but not src (extra from RQ1 output)
        extra_in_llm = llm_normalized - src_methods
        if extra_in_llm:
            # We need to exclude the actual zero-padded names from llm_oracle
            llm_extra_excluded[llm_path] = {llm_methods_map[norm] for norm in extra_in_llm}

    return src_excluded, llm_extra_excluded


# ---------------------------------------------------------------------------
# Step 4: Comment @Test by method name in a file
# ---------------------------------------------------------------------------

def comment_tests_in_file(filepath, excluded_method_names, add_dummy=True):
    """
    Comment @Test annotations for methods whose names are in excluded_method_names.
    Also adds test_nothing() dummy and disables separateClassLoader.
    Returns number of @Test annotations commented.
    """
    if not os.path.exists(filepath):
        print(f"  WARNING: File not found: {filepath}")
        return 0

    method_re = re.compile(r'public\s+void\s+(test\d+)\s*\(')

    # Normalize the excluded set so test05 matches test5 and vice versa
    normalized_excluded = {normalize_test_name(n) for n in excluded_method_names}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    content_str = "".join(lines)
    if 'void test_nothing(' in content_str:
        add_dummy = False

    commented = 0
    new_lines = []
    # We need two-pass: first find which @Test lines to comment
    # Strategy: walk forward, when we see @Test, peek ahead to find test method name

    i = 0
    dummy_added = False
    while i < len(lines):
        line = lines[i]

        # Disable separateClassLoader
        if 'separateClassLoader = true' in line:
            line = line.replace('separateClassLoader = true',
                                'separateClassLoader = false')

        # Add test_nothing() after class declaration
        if (add_dummy and not dummy_added
                and 'public class' in line and '_ESTest ' in line):
            new_lines.append(line)
            new_lines.append('  @Test(timeout = 4000)\n')
            new_lines.append(
                '  public void test_nothing()  throws Throwable  {}\n')
            dummy_added = True
            i += 1
            continue

        # Check if this is an @Test line
        if '@Test' in line:
            # Look ahead to find the method name
            method_name = None
            for j in range(i, min(i + 5, len(lines))):
                m = method_re.search(lines[j])
                if m:
                    method_name = m.group(1)
                    break
            
            # If it's one of the tests we want to exclude, comment it out
            # Normalize the method name to handle zero-padding differences
            if method_name and normalize_test_name(method_name) in normalized_excluded:
                line = '//' + line
                commented += 1

        new_lines.append(line)
        i += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    return commented


# ---------------------------------------------------------------------------
# Step 4: Map llm_oracle path to src and no_oracle paths
# ---------------------------------------------------------------------------

def get_corresponding_paths(llm_oracle_file, project_dir):
    """Get src and no_oracle paths corresponding to an llm_oracle file."""
    rel = os.path.relpath(llm_oracle_file, project_dir)
    parts = list(Path(rel).parts)

    for i, part in enumerate(parts):
        if part == 'llm_oracle':
            src_parts = parts[:i] + ['src'] + parts[i+1:]
            no_oracle_parts = parts[:i] + ['no_oracle'] + parts[i+1:]
            src_file = os.path.join(project_dir, *src_parts)
            no_oracle_file = os.path.join(project_dir, *no_oracle_parts)
            return src_file, no_oracle_file

    return None, None


# ---------------------------------------------------------------------------
# Step 5: Disable separateClassLoader in ALL test files
# ---------------------------------------------------------------------------

def disable_classloader_all(project_dir, already_processed):
    """Disable separateClassLoader and add test_nothing() in all _ESTest.java
    not already processed by comment_tests_in_file."""
    fixed = 0
    for root, dirs, files in os.walk(project_dir):
        # Skip .evosuite directories to avoid corrupting original EvoSuite output
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

            content_str = "".join(lines)
            dummy_already_exists = 'void test_nothing(' in content_str

            new_lines = []
            modified = False
            dummy_added = False
            for line in lines:
                # Disable separateClassLoader
                if 'separateClassLoader = true' in line:
                    line = line.replace('separateClassLoader = true',
                                        'separateClassLoader = false')
                    modified = True

                # Add test_nothing() after class declaration
                if (not dummy_already_exists and not dummy_added
                        and 'public class' in line and '_ESTest ' in line):
                    new_lines.append(line)
                    new_lines.append('  @Test(timeout = 4000)\n')
                    new_lines.append(
                        '  public void test_nothing()  throws Throwable  {}\n')
                    dummy_added = True
                    modified = True
                    continue

                new_lines.append(line)

            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# Step 6: Verify test counts match across all 3 suites
# ---------------------------------------------------------------------------

def count_active_tests(filepath):
    """Count @Test annotations that are NOT commented out in a file."""
    if not os.path.exists(filepath):
        return -1
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.lstrip()
            # Active @Test: starts with @Test (not //)
            if stripped.startswith('@Test'):
                count += 1
    return count


def verify_test_counts(project_dir):
    """
    Verify that each test file has the same number of active @Test methods
    across all 3 suites (src, llm_oracle, no_oracle).
    Returns (ok: bool, mismatches: list of dicts).
    """
    # Collect: suite_neutral_key -> {suite_name: (filepath, count)}
    file_map = {}  # key -> {'src': ..., 'llm_oracle': ..., 'no_oracle': ...}

    for root, dirs, files in os.walk(project_dir):
        if '.evosuite' in Path(root).parts:
            continue
        for name in files:
            if not name.endswith('_ESTest.java'):
                continue
            filepath = os.path.join(root, name)
            rel = os.path.relpath(filepath, project_dir)
            rel_parts = list(Path(rel).parts)

            suite_name = None
            for i, p in enumerate(rel_parts):
                if p in ('src', 'llm_oracle', 'no_oracle'):
                    suite_name = p
                    key = os.path.join(*rel_parts[:i], *rel_parts[i+1:])
                    break

            if not suite_name:
                continue

            active = count_active_tests(filepath)
            file_map.setdefault(key, {})[suite_name] = (filepath, active)

    mismatches = []
    suite_totals = defaultdict(int)  # suite_name -> total active @Test
    for key, suites in sorted(file_map.items()):
        counts = {s: info[1] for s, info in suites.items()}
        for s, cnt in counts.items():
            suite_totals[s] += cnt
        unique_counts = set(counts.values())
        # Skip files that only exist in one suite
        if len(suites) < 2:
            continue
        if len(unique_counts) > 1:
            mismatches.append({'file': key, 'counts': counts})

    return len(mismatches) == 0, mismatches, dict(suite_totals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_test_num(name):
    m = re.search(r'\d+', name)
    return int(m.group()) if m else -1

def copy_input_to_output(input_dir, output_dir):
    """Copy project from input_dir to output_dir, skipping target/ directories."""
    if os.path.exists(output_dir):
        print(f"  Output already exists, removing: {output_dir}")
        shutil.rmtree(output_dir)

    def ignore_targets(src, names):
        ignored = set()
        for name in names:
            full = os.path.join(src, name)
            if name == 'target' and os.path.isdir(full):
                ignored.add(name)
        return ignored

    print(f"  Copying {input_dir}")
    print(f"       -> {output_dir}")
    shutil.copytree(input_dir, output_dir, ignore=ignore_targets)
    print(f"  Copy done.")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare test suites for PIT mutation testing')
    parser.add_argument('--input_dir', default=None,
                        help='(Optional) Source to copy from (data/RQ2/input/<project>). '
                             'If given, project_dir will be populated from here first.')
    parser.add_argument('--project_dir', required=True,
                        help='Path to output project dir (data/RQ2/output/<project>)')
    parser.add_argument('--surefire_base', required=True, nargs='+',
                        help='Path(s) to surefire reports '
                             '(e.g. data/RQ1/output/<project>/injected_assertion and/or data/RQ2/output/<project>)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only report, do not modify files')
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    surefire_bases = [os.path.abspath(p) for p in args.surefire_base]

    # --- Step 0: Copy input → output ---
    if args.input_dir:
        input_dir = os.path.abspath(args.input_dir)
        if not args.dry_run:
            copy_input_to_output(input_dir, project_dir)
        else:
            print(f"[DRY RUN] Would copy {input_dir} -> {project_dir}")

    # --- Step 1: Tce ---
    tce_excluded = find_tce_tests(project_dir)
    tce_count = sum(len(v) for v in tce_excluded.values())

    tfp_excluded = find_tfp_tests(surefire_bases, project_dir)
    tfp_count = sum(len(v) for v in tfp_excluded.values())

    exc_excluded, llm_extra_excluded = find_exception_tests(project_dir)
    exc_count = sum(len(v) for v in exc_excluded.values())
    llm_extra_count = sum(len(v) for v in llm_extra_excluded.values())

    # --- Merge ---
    all_excluded_llm = defaultdict(set)  # llm_oracle-keyed
    for f, names in tce_excluded.items():
        all_excluded_llm[f].update(names)
    for f, names in tfp_excluded.items():
        all_excluded_llm[f].update(names)

    total_tce_tfp = sum(len(v) for v in all_excluded_llm.values())
    total_excluded = total_tce_tfp + exc_count + llm_extra_count

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
        rel = os.path.relpath(src_file, project_dir)
        parts = list(Path(rel).parts)
        for i, p in enumerate(parts):
            if p == 'src':
                no_parts = parts[:i] + ['no_oracle'] + parts[i+1:]
                no_oracle_file = os.path.join(project_dir, *no_parts)
                no_exclusions[no_oracle_file].update(methods)

                llm_parts = parts[:i] + ['llm_oracle'] + parts[i+1:]
                llm_oracle_file = os.path.join(project_dir, *llm_parts)
                llm_exclusions[llm_oracle_file].update(methods)
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

    cl_fixed = disable_classloader_all(project_dir, all_processed_files)

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


if __name__ == '__main__':
    main()

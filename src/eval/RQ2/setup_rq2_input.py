import os
import shutil
import argparse
from pathlib import Path

EXCLUDE_DIRS = {'target', '.evosuite', '.git', 'infer_input', 'toga_output', 'results'}


# Tìm các module Maven có chứa file *_ESTest.java (test do EvoSuite sinh).
def find_estest_modules(project_dir):
    seen = set()
    modules = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        parts = Path(root).parts
        if 'src' not in parts or 'test' not in parts:
            continue
        for f in files:
            if f.endswith('_ESTest.java'):
                rel = os.path.relpath(root, project_dir)
                path_parts = rel.split(os.sep)
                try:
                    idx = path_parts.index('src')
                    module_rel = os.sep.join(path_parts[:idx])
                except ValueError:
                    continue
                if module_rel not in seen:
                    seen.add(module_rel)
                    modules.append((module_rel, os.path.join(project_dir, module_rel)))
                break
    return modules


# Copy file test sang đích, đồng thời comment (// ) mọi dòng bắt đầu bằng "assert".
def copy_and_comment_asserts(src_path, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    with open(dest_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip().startswith('assert'):
                line = '//' + line
            f.write(line)


# Tạo suite no_oracle: copy test EvoSuite gốc và comment toàn bộ assertion.
def create_no_oracle(module_path, module_rel, rq2_out):
    src_test = os.path.join(module_path, 'src', 'test', 'java')
    if not os.path.exists(src_test):
        return 0
    count = 0
    for root, dirs, files in os.walk(src_test):
        for f in files:
            if '_ESTest' in f and f.endswith('.java'):
                src_file = os.path.join(root, f)
                rel = os.path.relpath(src_file, src_test)
                dest = os.path.join(rq2_out, module_rel, 'no_oracle', 'test', 'java', rel)
                if f.endswith('_ESTest.java'):
                    copy_and_comment_asserts(src_file, dest)
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(src_file, dest)
                count += 1
    return count


# Tạo suite llm_oracle: copy test đã được pipeline AssertGen inject assertion từ RQ1/output.
def create_llm_oracle(module_rel, injected_dir, rq2_out):
    inj_test = os.path.join(injected_dir, module_rel, 'src', 'test', 'java')
    if not os.path.exists(inj_test):
        return 0
    count = 0
    for root, dirs, files in os.walk(inj_test):
        for f in files:
            if '_ESTest' in f and f.endswith('.java'):
                src_file = os.path.join(root, f)
                rel = os.path.relpath(src_file, inj_test)
                dest = os.path.join(rq2_out, module_rel, 'llm_oracle', 'test', 'java', rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src_file, dest)
                count += 1
    return count


# Copy pom.xml và pit.sh từ TOGLL reference repo để dùng cấu hình PIT chuẩn.
def copy_pit_and_pom_from_togll(module_rel, rq2_out, togll_root):
    if not togll_root:
        return
    src_module = os.path.join(togll_root, module_rel) if module_rel else togll_root
    if not os.path.isdir(src_module):
        return
    dest_dir = os.path.join(rq2_out, module_rel) if module_rel else rq2_out
    os.makedirs(dest_dir, exist_ok=True)
    for fname in ('pom.xml', 'pit.sh'):
        src = os.path.join(src_module, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, fname))


# Entry point: copy RQ1/input -> RQ2/output, dựng đủ 3 suite (src, llm_oracle, no_oracle) cho mỗi module.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq1_input', required=True)
    parser.add_argument('--rq1_injected', required=True)
    parser.add_argument('--rq2_output', required=True)
    parser.add_argument('--togll_root')
    args = parser.parse_args()

    rq1_input = os.path.abspath(args.rq1_input)
    rq1_injected = os.path.abspath(args.rq1_injected)
    rq2_out = os.path.abspath(args.rq2_output)
    togll_root = os.path.abspath(args.togll_root) if args.togll_root else None

    if os.path.exists(rq2_out):
        shutil.rmtree(rq2_out)
    shutil.copytree(rq1_input, rq2_out, ignore=shutil.ignore_patterns('target', '.evosuite', '.git'))

    modules = find_estest_modules(rq2_out)
    print(f"Found {len(modules)} modules")
    for module_rel, module_path in sorted(modules):
        n_llm = create_llm_oracle(module_rel, rq1_injected, rq2_out)
        n_no = create_no_oracle(module_path, module_rel, rq2_out)
        print(f"  {module_rel or '(root)'}: llm_oracle={n_llm} no_oracle={n_no}")
        copy_pit_and_pom_from_togll(module_rel, rq2_out, togll_root)
    print(f"Done: {rq2_out}")


if __name__ == '__main__':
    main()

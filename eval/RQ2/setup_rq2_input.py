"""Setup RQ2 input from RQ1 output (like async-http-client structure).

Creates data/RQ2/input/<project> with src, llm_oracle, no_oracle per module.
Requires: RQ1 input (project source), RQ1 injected_assertion output.

Usage:
    python eval/RQ2/setup_rq2_input.py \
        --rq1_input data/RQ1/input/commons-weaver-2.0-src \
        --rq1_injected data/RQ1/output/commons-weaver-2.0-src/injected_assertion \
        --rq2_output data/RQ2/input/commons-weaver-2.0-src
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

assert_re = re.compile(r"assert\w*\(.*\)")
EXCLUDE_DIRS = {'target', '.evosuite', '.git', 'infer_input', 'toga_output', 'results'}


def find_estest_modules(project_dir):
    """Find modules containing _ESTest.java under src/test."""
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


def copy_and_comment_asserts(src_path, dest_path):
    """Copy file and comment out assert lines."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    with open(dest_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip().startswith('assert'):
                line = '//' + line
            f.write(line)


def create_no_oracle(module_path, module_rel, rq2_out):
    """Create no_oracle from src. Scaffolding: copy as-is. _ESTest: comment asserts."""
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


def create_llm_oracle(module_path, module_rel, injected_dir, rq2_out):
    """Create llm_oracle from RQ1 injected_assertion. Copy _ESTest + _ESTest_scaffolding."""
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


def copy_pit_and_pom_from_togll(module_rel, rq2_out, togll_root):
    """Copy pit.sh và pom.xml từ TOGLL RQ4 (artifacts_with_es_togll_tests) nếu có."""
    if not togll_root:
        return
    src_module = os.path.join(togll_root, module_rel) if module_rel else togll_root
    if not os.path.isdir(src_module):
        return

    # pom.xml
    pom_src = os.path.join(src_module, 'pom.xml')
    pom_dst = os.path.join(rq2_out, module_rel, 'pom.xml') if module_rel else os.path.join(rq2_out, 'pom.xml')
    if os.path.exists(pom_src):
        os.makedirs(os.path.dirname(pom_dst), exist_ok=True)
        shutil.copy2(pom_src, pom_dst)

    # pit.sh (giả định ở ngay dưới module)
    pit_src = os.path.join(src_module, 'pit.sh')
    pit_dst = os.path.join(rq2_out, module_rel, 'pit.sh') if module_rel else os.path.join(rq2_out, 'pit.sh')
    if os.path.exists(pit_src):
        os.makedirs(os.path.dirname(pit_dst), exist_ok=True)
        shutil.copy2(pit_src, pit_dst)


def write_pit_sh(module_path, module_rel, rq2_out, target_pkg):
    """Write pit.sh for module (matches async-http-client/TOGLL: redirect PIT to .txt, no cd, no -q)."""
    pit_dir = os.path.join(rq2_out, module_rel)
    os.makedirs(pit_dir, exist_ok=True)
    content = '''#!/bin/bash
echo "========================================="
echo "Running Baseline (src)..."
echo "========================================="
mvn clean test -PtestID -Dtest.dir=src -PtargetID -Dtarget.dir=target || true
mvn -Dhttps.protocols=TLSv1.2 -PtestID -Dtest.dir=src -PtargetID -Dtarget.dir=target org.pitest:pitest-maven:1.9.8:mutationCoverage > es_pit.txt

echo "========================================="
echo "Running LLM Oracle (llm_oracle)..."
echo "========================================="
mvn clean test -PtestID -Dtest.dir=llm_oracle -PtargetID -Dtarget.dir=llm_oracle/target || true
mvn -Dhttps.protocols=TLSv1.2 -PtestID -Dtest.dir=llm_oracle -PtargetID -Dtarget.dir=llm_oracle/target org.pitest:pitest-maven:1.9.8:mutationCoverage > llm_oracle_pit.txt

echo "========================================="
echo "Running No Oracle (no_oracle)..."
echo "========================================="
mvn clean test -PtestID -Dtest.dir=no_oracle -PtargetID -Dtarget.dir=no_oracle/target || true
mvn -Dhttps.protocols=TLSv1.2 -PtestID -Dtest.dir=no_oracle -PtargetID -Dtarget.dir=no_oracle/target org.pitest:pitest-maven:1.9.8:mutationCoverage > no_oracle_pit.txt
'''
    pit_file = os.path.join(pit_dir, 'pit.sh')
    with open(pit_file, 'w') as f:
        f.write(content)
    os.chmod(pit_file, 0o755)


def _infer_target_package(module_rel):
    """Infer PIT targetClasses from module path (commons-weaver convention)."""
    m = module_rel.replace(os.sep, '.').replace('/', '.')
    if m.startswith('modules.'):
        m = m.split('.', 1)[1].split('.')[0]
    if m in ('processor',):
        return "org.apache.commons.weaver*"
    return f"org.apache.commons.weaver.{m}*" if m else "org.apache.commons.weaver*"


def add_pom_profiles(module_path, rq2_out, module_rel):
    """Add testID/targetID, testSourceDirectory, and pitest-maven plugin (match TOGLL)."""
    pom_src = os.path.join(rq2_out, module_rel, 'pom.xml')
    if not os.path.exists(pom_src):
        return False
    with open(pom_src, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    if 'test.dir' in content and 'pitest-maven' in content:
        return True
    if '<properties>' in content and 'test.dir' not in content:
        content = content.replace(
            '<properties>',
            '<properties>\n    <test.dir>src</test.dir>\n    <target.dir>target</target.dir>',
            1
        )
    elif '<properties>' not in content:
        content = content.replace(
            '<modelVersion>',
            '<properties>\n    <test.dir>src</test.dir>\n    <target.dir>target</target.dir>\n  </properties>\n  <modelVersion>',
            1
        )
    build_insert = '\n    <testSourceDirectory>${project.basedir}/${test.dir}/test/java</testSourceDirectory>\n    <directory>${project.basedir}/${target.dir}/</directory>'
    if '<build>' in content and 'testSourceDirectory' not in content:
        content = content.replace('<build>', '<build>' + build_insert, 1)
    if 'pitest-maven' not in content:
        target_pkg = _infer_target_package(module_rel)
        pitest_plugin = f'''
      <plugin>
        <groupId>org.pitest</groupId>
        <artifactId>pitest-maven</artifactId>
        <version>1.9.8</version>
        <configuration>
          <threads>4</threads>
          <timeoutConstant>8000</timeoutConstant>
          <mutators><mutator>STRONGER</mutator></mutators>
          <avoidCallsTo>
            <avoidCallsTo>java.*</avoidCallsTo>
            <avoidCallsTo>org.apache.log4j</avoidCallsTo>
            <avoidCallsTo>org.slf4j.*</avoidCallsTo>
            <avoidCallsTo>org.apache.commons.logging</avoidCallsTo>
          </avoidCallsTo>
          <verbose>false</verbose>
          <targetClasses><param>{target_pkg}</param></targetClasses>
          <targetTests><param>org.apache.commons.weaver.*_ESTest*</param></targetTests>
          <fullMutationMatrix>true</fullMutationMatrix>
          <exportLineCoverage>true</exportLineCoverage>
          <outputFormats>XML</outputFormats>
        </configuration>
      </plugin>'''
        if '<plugins>' in content:
            content = content.replace('<plugins>', '<plugins>' + pitest_plugin, 1)
    profiles = '''
  <profiles>
    <profile>
      <id>testID</id>
      <activation><property><name>test.dir</name></property></activation>
    </profile>
    <profile>
      <id>targetID</id>
      <activation><property><name>target.dir</name></property></activation>
    </profile>
  </profiles>
'''
    if '</project>' in content and 'testID' not in content:
        content = content.replace('</project>', profiles + '\n</project>')
    with open(pom_src, 'w') as f:
        f.write(content)
    return True


def main():
    parser = argparse.ArgumentParser(description='Setup RQ2 input from RQ1')
    parser.add_argument('--rq1_input', required=True, help='RQ1 input (project source)')
    parser.add_argument('--rq1_injected', required=True, help='RQ1 injected_assertion output')
    parser.add_argument('--rq2_output', required=True, help='RQ2 input output dir')
    parser.add_argument('--togll_root', help='Path tới togll/RQ4/artifacts_with_es_togll_tests/<project> để copy pit/pom')
    args = parser.parse_args()

    rq1_input = os.path.abspath(args.rq1_input)
    rq1_injected = os.path.abspath(args.rq1_injected)
    rq2_out = os.path.abspath(args.rq2_output)
    togll_root = os.path.abspath(args.togll_root) if args.togll_root else None

    if os.path.exists(rq2_out):
        print(f"Removing existing {rq2_out}")
        shutil.rmtree(rq2_out)
    print(f"Copying {rq1_input} -> {rq2_out}")
    shutil.copytree(rq1_input, rq2_out, ignore=shutil.ignore_patterns('target', '.evosuite', '.git'))

    modules = find_estest_modules(rq2_out)
    print(f"Found {len(modules)} modules with _ESTest.java")

    for module_rel, module_path in sorted(modules):
        print(f"\n  {module_rel or '(root)'}:")
        n = create_llm_oracle(module_path, module_rel, rq2_out, rq2_out)
        print(f"    llm_oracle: {n} files")
        n = create_no_oracle(module_path, module_rel, rq2_out)
        print(f"    no_oracle: {n} files")
        # Nếu cung cấp togll_root, copy pit.sh + pom.xml từ TOGLL RQ4.
        copy_pit_and_pom_from_togll(module_rel, rq2_out, togll_root)

    print(f"\nDone. RQ2 input: {rq2_out}")
    print("Next: python eval/RQ2/prepare_for_pit.py \\")
    print(f"  --input_dir {rq2_out} \\")
    print(f"  --project_dir data/RQ2/output/commons-weaver-2.0-src \\")
    print(f"  --surefire_base data/RQ1/output/commons-weaver-2.0-src/injected_assertion")


if __name__ == '__main__':
    main()

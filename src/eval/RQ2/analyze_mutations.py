"""analyze_mutations.py

Usage:
    python eval/RQ2/analyze_mutations.py --data_dir data/RQ2/output --projects async-http-client
"""

import os
import re
import sys
import json
import argparse
from xml.dom import minidom
from pathlib import Path
from collections import defaultdict


def find_mutations_xml(base_dir):
    pit_dir = os.path.join(base_dir, 'pit-reports')
    if not os.path.exists(pit_dir):
        return None
    for subdir in sorted(os.listdir(pit_dir), reverse=True):
        candidate = os.path.join(pit_dir, subdir, 'mutations.xml')
        if os.path.exists(candidate):
            return candidate

    # Also check directly
    direct = os.path.join(pit_dir, 'mutations.xml')
    if os.path.exists(direct):
        return direct

    return None


def analyze_project(project_dir, project_name):
    """
    Analyze mutation testing results for a single project.
    Walks the project directory to find pit-reports in each module's target dirs.
    """
    results = {
        'project': project_name,
        'total_mutants_with_coverage': 0,
        'implicit_detections': 0,
        'es_killed': 0,
        'llm_killed': 0,
        'es_unique': 0,
        'llm_unique': 0,
        'modules': []
    }

    modules_found = set()
    for root, dirs, files in os.walk(project_dir):
        if 'pit-reports' in dirs:
            # Determine module path relative to project_dir
            rel = os.path.relpath(root, project_dir)
            # Strip target/ suffix to get module path
            parts = Path(rel).parts
            if 'target' in parts:
                idx = parts.index('target')
                # Check if this is under llm_oracle or no_oracle
                if idx > 0 and parts[idx-1] in ('llm_oracle', 'no_oracle'):
                    module = str(Path(*parts[:idx-1])) if idx > 1 else '.'
                else:
                    module = str(Path(*parts[:idx])) if idx > 0 else '.'
                modules_found.add(module)

    if not modules_found:
        print(f"  WARNING: No pit-reports found in {project_dir}")
        return results

    for module in sorted(modules_found):
        module_path = os.path.join(project_dir, module) if module != '.' \
            else project_dir

        # Find mutations.xml for each suite
        es_xml = find_mutations_xml(os.path.join(module_path, 'target'))
        llm_xml = find_mutations_xml(
            os.path.join(module_path, 'llm_oracle', 'target'))
        no_xml = find_mutations_xml(
            os.path.join(module_path, 'no_oracle', 'target'))

        if not all([es_xml, llm_xml, no_xml]):
            missing = []
            if not es_xml:
                missing.append('es')
            if not llm_xml:
                missing.append('llm_oracle')
            if not no_xml:
                missing.append('no_oracle')
            print(f"  WARNING: Module '{module}' missing: {missing}")
            continue

        print(f"  Module: {module}")
        module_result = analyze_module(es_xml, llm_xml, no_xml, module)
        results['modules'].append(module_result)

        results['total_mutants_with_coverage'] += \
            module_result['mutants_with_coverage']
        results['implicit_detections'] += module_result['implicit']
        results['es_killed'] += module_result['es_killed']
        results['llm_killed'] += module_result['llm_killed']
        results['es_unique'] += module_result['es_unique']
        results['llm_unique'] += module_result['llm_unique']

    # Compute mutation scores
    covered = results['total_mutants_with_coverage']
    implicit = results['implicit_detections']
    non_implicit = covered - implicit  # mutants that no_oracle didn't kill

    results['MS_es'] = ((results['es_killed'] + implicit) / covered * 100
                        if covered > 0 else 0.0)
    results['MS_llm'] = ((results['llm_killed'] + implicit) / covered * 100
                         if covered > 0 else 0.0)
    results['MS_no_oracle'] = (implicit / covered * 100
                               if covered > 0 else 0.0)

    return results


def analyze_module(es_xml_path, llm_xml_path, no_xml_path, module_name):
    """Compare mutations across 3 suites for one module."""
    es_doc = minidom.parse(es_xml_path)
    llm_doc = minidom.parse(llm_xml_path)
    no_doc = minidom.parse(no_xml_path)

    mutations_es = es_doc.getElementsByTagName('mutation')
    mutations_llm = llm_doc.getElementsByTagName('mutation')
    mutations_no = no_doc.getElementsByTagName('mutation')

    if len(mutations_es) != len(mutations_llm):
        print(f"    WARNING: Mutant count mismatch ES({len(mutations_es)}) "
              f"vs LLM({len(mutations_llm)})")
    if len(mutations_es) != len(mutations_no):
        print(f"    WARNING: Mutant count mismatch ES({len(mutations_es)}) "
              f"vs NO({len(mutations_no)})")

    total_mutants = len(mutations_es)
    mutants_with_coverage = 0
    implicit = 0
    es_killed = 0
    llm_killed = 0
    es_unique = 0
    llm_unique = 0

    for m_es, m_llm, m_no in zip(mutations_es, mutations_llm, mutations_no):
        status_no = m_no.attributes['status'].value
        detected_es = m_es.attributes['detected'].value == 'true'
        detected_llm = m_llm.attributes['detected'].value == 'true'
        detected_no = m_no.attributes['detected'].value == 'true'

        if 'NO_COVERAGE' in status_no:
            continue
        mutants_with_coverage += 1
        if detected_no:
            implicit += 1
            continue
        if detected_es:
            es_killed += 1
        if detected_llm:
            llm_killed += 1
        if detected_es and not detected_llm:
            es_unique += 1
        if detected_llm and not detected_es:
            llm_unique += 1

    module_result = {
        'module': module_name,
        'total_mutants': total_mutants,
        'mutants_with_coverage': mutants_with_coverage,
        'implicit': implicit,
        'es_killed': es_killed,
        'llm_killed': llm_killed,
        'es_unique': es_unique,
        'llm_unique': llm_unique,
    }

    print(f"    Mutants: {total_mutants} total, {mutants_with_coverage} covered")
    print(f"    Implicit: {implicit}")
    print(f"    ES killed: {es_killed} (unique: {es_unique})")
    print(f"    LLM killed: {llm_killed} (unique: {llm_unique})")

    return module_result


def main():
    parser = argparse.ArgumentParser(
        description='Analyze PIT mutation testing results for RQ2')
    parser.add_argument('--data_dir', required=True,
                        help='Path to data/RQ2')
    parser.add_argument('--projects', nargs='*', default=None,
                        help='Specific project names (default: all)')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    if args.projects:
        projects = args.projects
    else:
        projects = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
               and os.path.exists(os.path.join(data_dir, d, 'pit.sh'))
        ])

    all_results = []
    for project in projects:
        project_dir = os.path.join(data_dir, project)
        print("=" * 60)
        print(f"Project: {project}")
        print("=" * 60)

        result = analyze_project(project_dir, project)
        all_results.append(result)

        # Save per-project result inside the project folder.
        project_output_file = os.path.join(project_dir, 'rq2_results.json')
        with open(project_output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved project result to: {project_output_file}")

        covered = result['total_mutants_with_coverage']
        if covered > 0:
            print(f"\n  --- Mutation Scores ---")
            print(f"  MS (no oracle):  {result['MS_no_oracle']:.2f}%")
            print(f"  MS (EvoSuite):   {result['MS_es']:.2f}%")
            print(f"  MS (LLM):        {result['MS_llm']:.2f}%")
            print(f"  ES unique kills: {result['es_unique']}")
            print(f"  LLM unique kills: {result['llm_unique']}")
        print()

    # Summary table
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Project':<30} {'Covered':>8} {'Implicit':>8} "
          f"{'ES':>6} {'LLM':>6} {'MS_es':>7} {'MS_llm':>7}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['project']:<30} "
              f"{r['total_mutants_with_coverage']:>8} "
              f"{r['implicit_detections']:>8} "
              f"{r['es_killed']:>6} "
              f"{r['llm_killed']:>6} "
              f"{r['MS_es']:>6.2f}% "
              f"{r['MS_llm']:>6.2f}%")

    print("\nPer-project results saved in each project folder.")


if __name__ == '__main__':
    main()

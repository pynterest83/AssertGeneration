import os
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
    direct = os.path.join(pit_dir, 'mutations.xml')
    return direct if os.path.exists(direct) else None

# analyze single project by analyze internal modules
def analyze_project(project_dir, project_name):
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
            rel = os.path.relpath(root, project_dir)
            parts = Path(rel).parts
            if 'target' in parts:
                idx = parts.index('target')
                if idx > 0 and parts[idx-1] in ('llm_oracle', 'no_oracle'):
                    module = str(Path(*parts[:idx-1])) if idx > 1 else '.'
                else:
                    module = str(Path(*parts[:idx])) if idx > 0 else '.'
                modules_found.add(module)

    if not modules_found:
        print(f"  WARNING: No pit-reports found in {project_dir}")
        return results

    for module in sorted(modules_found):
        module_path = os.path.join(project_dir, module) if module != '.' else project_dir

        es_xml = find_mutations_xml(os.path.join(module_path, 'target'))
        llm_xml = find_mutations_xml(os.path.join(module_path, 'llm_oracle', 'target'))
        no_xml = find_mutations_xml(os.path.join(module_path, 'no_oracle', 'target'))

        if not all([es_xml, llm_xml, no_xml]):
            missing = [s for s, x in [('es', es_xml), ('llm_oracle', llm_xml), ('no_oracle', no_xml)] if not x]
            print(f"  WARNING: Module '{module}' missing: {missing}")
            continue

        print(f"  Module: {module}")
        module_result = analyze_module(es_xml, llm_xml, no_xml, module)
        results['modules'].append(module_result)
        results['total_mutants_with_coverage'] += module_result['mutants_with_coverage']
        results['implicit_detections'] += module_result['implicit']
        results['es_killed'] += module_result['es_killed']
        results['llm_killed'] += module_result['llm_killed']
        results['es_unique'] += module_result['es_unique']
        results['llm_unique'] += module_result['llm_unique']

    covered = results['total_mutants_with_coverage']
    implicit = results['implicit_detections']
    results['MS_es'] = (results['es_killed'] + implicit) / covered * 100 if covered else 0.0
    results['MS_llm'] = (results['llm_killed'] + implicit) / covered * 100 if covered else 0.0
    results['MS_no_oracle'] = implicit / covered * 100 if covered else 0.0

    return results


def _child_text(elem, tag):
    nodes = elem.getElementsByTagName(tag)
    if nodes and nodes[0].firstChild:
        return nodes[0].firstChild.nodeValue
    return ''

# get detail info of a mutation
def _mutation_key(m):
    return (
        _child_text(m, 'mutatedClass'),
        _child_text(m, 'mutatedMethod'),
        _child_text(m, 'mutatedMethodDesc'),
        _child_text(m, 'lineNumber'),
        _child_text(m, 'mutator'),
        _child_text(m, 'index'),
    )

# get status if killed, no_coverage, etc. and whether detected (killed) or not
def _mut_info(m):
    return {
        'status': m.attributes['status'].value,
        'detected': m.attributes['detected'].value == 'true',
    }


_NO_COVERAGE = {'status': 'NO_COVERAGE', 'detected': False}

# Analyze a single module by comparing the mutation results from ES, LLM, and NO oracles
def analyze_module(es_xml_path, llm_xml_path, no_xml_path, module_name):
    es_map = {_mutation_key(m): _mut_info(m) for m in minidom.parse(es_xml_path).getElementsByTagName('mutation')}
    llm_map = {_mutation_key(m): _mut_info(m) for m in minidom.parse(llm_xml_path).getElementsByTagName('mutation')}
    no_map = {_mutation_key(m): _mut_info(m) for m in minidom.parse(no_xml_path).getElementsByTagName('mutation')}

    if len(es_map) != len(llm_map):
        print(f"    WARNING: Mutant count mismatch ES({len(es_map)}) vs LLM({len(llm_map)})")
    if len(es_map) != len(no_map):
        print(f"    WARNING: Mutant count mismatch ES({len(es_map)}) vs NO({len(no_map)})")

    all_keys = set(es_map) | set(llm_map) | set(no_map)
    total_mutants = len(all_keys)
    mutants_with_coverage = 0
    implicit = 0
    es_killed = 0
    llm_killed = 0
    es_unique = 0
    llm_unique = 0

    for key in all_keys:
        m_es = es_map.get(key, _NO_COVERAGE)
        m_llm = llm_map.get(key, _NO_COVERAGE)
        m_no = no_map.get(key, _NO_COVERAGE)

        if m_no['status'] == 'NO_COVERAGE':
            continue
        mutants_with_coverage += 1
        if m_no['detected']:
            implicit += 1
            continue
        if m_es['detected']:
            es_killed += 1
        if m_llm['detected']:
            llm_killed += 1
        if m_es['detected'] and not m_llm['detected']:
            es_unique += 1
        if m_llm['detected'] and not m_es['detected']:
            llm_unique += 1

    print(f"    Mutants: {total_mutants} total, {mutants_with_coverage} covered")
    print(f"    Implicit: {implicit} | ES: {es_killed} (u:{es_unique}) | LLM: {llm_killed} (u:{llm_unique})")

    return {
        'module': module_name,
        'total_mutants': total_mutants,
        'mutants_with_coverage': mutants_with_coverage,
        'implicit': implicit,
        'es_killed': es_killed,
        'llm_killed': llm_killed,
        'es_unique': es_unique,
        'llm_unique': llm_unique,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze PIT mutation testing results for RQ2')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--projects', nargs='*', default=None)
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

        project_output_file = os.path.join(project_dir, 'rq2_results.json')
        with open(project_output_file, 'w') as f:
            json.dump(result, f, indent=2)

        covered = result['total_mutants_with_coverage']
        if covered > 0:
            print(f"  MS_no={result['MS_no_oracle']:.2f}%  MS_es={result['MS_es']:.2f}%  MS_llm={result['MS_llm']:.2f}%")
            print(f"  ES unique: {result['es_unique']}  LLM unique: {result['llm_unique']}")
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Project':<30} {'Covered':>8} {'Implicit':>8} {'ES':>6} {'LLM':>6} {'MS_es':>7} {'MS_llm':>7}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['project']:<30} "
              f"{r['total_mutants_with_coverage']:>8} "
              f"{r['implicit_detections']:>8} "
              f"{r['es_killed']:>6} "
              f"{r['llm_killed']:>6} "
              f"{r['MS_es']:>6.2f}% "
              f"{r['MS_llm']:>6.2f}%")


if __name__ == '__main__':
    main()

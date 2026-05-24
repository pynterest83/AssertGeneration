"""Inject predicted assertions into Python test files.

Strategy (ported from archieved/extension/backend/scripts/python_injector.py):
  1. Read oracle_preds.csv, group rows by file.
  2. For tests with multiple assertN, keep the highest N (last assert).
  3. Build new method strings: test_prefix + indented assertion.
  4. AST-parse the file, remove the old test methods (NodeTransformer), unparse.
  5. Append new methods at the end with a marker comment.

Caveats:
  - ast.unparse loses comments. This is a known limitation.
  - If the test prefix already contains the latest assert, behavior is correct
    because the prefix for assertN includes asserts 1..N-1.
"""
import ast
import csv
import os


def _base_test_name(test_name: str) -> str:
    if "_assert" in test_name and test_name.split("_assert")[-1].isdigit():
        return test_name.rsplit("_assert", 1)[0]
    return test_name


def _indentation(test_prefix: str) -> int:
    for line in reversed(test_prefix.split("\n")):
        if line.strip() and not line.strip().startswith("def "):
            return len(line) - len(line.lstrip())
    return 4


def _find_file(repo_dir: str, filename: str) -> str | None:
    direct = os.path.join(repo_dir, filename)
    if os.path.isfile(direct):
        return direct
    bare = os.path.basename(filename)
    for root, _, files in os.walk(repo_dir):
        if bare in files:
            return os.path.join(root, bare)
    return None


def inject_tests(repo_dir: str, preds_csv: str) -> list[str]:
    with open(preds_csv, "r", encoding="utf-8") as f:
        preds = list(csv.DictReader(f))

    file_map: dict[str, list[dict]] = {}
    for row in preds:
        file_map.setdefault(row["file_path"], []).append(row)

    modified_files: list[str] = []

    for fname, rows in file_map.items():
        original_path = _find_file(repo_dir, fname)
        if not original_path:
            print(f"Warning: file not found in repo: {fname}")
            continue

        with open(original_path, "r", encoding="utf-8") as f:
            code = f.read()

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"Warning: syntax error in {original_path}: {e}")
            continue

        # Keep only highest _assertN per base test
        test_groups: dict[str, dict] = {}
        for row in rows:
            name = row["test_name"]
            base = _base_test_name(name)
            if base not in test_groups:
                test_groups[base] = row
            else:
                current_name = test_groups[base]["test_name"]
                if "_assert" in name:
                    curr_idx = int(current_name.split("_assert")[-1]) if "_assert" in current_name else 1
                    new_idx = int(name.split("_assert")[-1])
                    if new_idx > curr_idx:
                        test_groups[base] = row

        base_names_to_replace = set(test_groups.keys())

        new_methods_code: list[str] = []
        for row in test_groups.values():
            prefix = row.get("test_prefix", "")
            assertion = row.get("assert_pred", "")
            if not prefix or not assertion:
                continue
            ind_str = " " * _indentation(prefix)
            assertion_lines = assertion.strip().split("\n")
            indented_assertion = "\n".join(ind_str + al.lstrip() for al in assertion_lines)
            new_methods_code.append(f"{prefix}\n{indented_assertion}")

        class _Remover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name in base_names_to_replace:
                    return None
                self.generic_visit(node)
                return node

            visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        tree = _Remover().visit(tree)
        ast.fix_missing_locations(tree)
        clean_code = ast.unparse(tree)

        final_code = clean_code + "\n\n# --- AUTOMATICALLY INJECTED TESTS ---\n\n"
        for method_str in new_methods_code:
            final_code += method_str + "\n\n"

        with open(original_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        modified_files.append(original_path)
        print(f"Injected {len(new_methods_code)} test(s) into {original_path}")

    return modified_files

"""Inject predicted assertions into Java test files.

Strategy (string-based, no AST so comments + formatting are preserved):
  1. Read oracle_preds.csv, group rows by file.
  2. For tests with multiple _assertN, keep the highest N (matches Python injector).
  3. For each test method (@Test ... methodName(...) { ... }):
     - Locate the method by name via regex + brace counting.
     - Replace its body with test_prefix + indented assertion.
  4. Write the modified file.

Assumption: dataset is TOGA-style — i.e. test_prefix == the method body up to the
target assertion location, so replacing the body wholesale is semantically
correct. If the original method had additional code after the assert, it will
be lost (this matches Python injector behavior).
"""
import csv
import os
import re


_TEST_METHOD_PATTERN = re.compile(
    r"@Test[^\n]*\n\s*(?:@[^\n]+\n\s*)*"
    r"(?:public\s+)?(?:void|[\w<>\[\]]+)\s+"
    r"(?P<name>test\w*|should\w*|\w+Test\w*)\s*\([^)]*\)"
    r"(?:\s*throws\s+[\w,\s]+)?\s*\{",
    re.MULTILINE,
)


def _base_test_name(test_name: str) -> str:
    if "_assert" in test_name and test_name.split("_assert")[-1].isdigit():
        return test_name.rsplit("_assert", 1)[0]
    return test_name


def _find_file(repo_dir: str, filename: str) -> str | None:
    direct = os.path.join(repo_dir, filename)
    if os.path.isfile(direct):
        return direct
    bare = os.path.basename(filename)
    for root, _, files in os.walk(repo_dir):
        if bare in files:
            return os.path.join(root, bare)
    return None


def _detect_body_indent(method_body: str) -> str:
    """Pick the smallest non-zero leading-whitespace of any non-blank line."""
    indents = []
    for line in method_body.splitlines():
        if not line.strip():
            continue
        stripped_len = len(line) - len(line.lstrip())
        if stripped_len > 0:
            indents.append(stripped_len)
    return " " * (min(indents) if indents else 8)


def _replace_method_body(source: str, method_name: str,
                         new_body_content: str) -> tuple[str, bool]:
    """Replace the body between { and the matching } for a given @Test method.

    Returns (new_source, replaced_flag).
    """
    for match in _TEST_METHOD_PATTERN.finditer(source):
        if match.group("name") != method_name:
            continue
        # Detect the method's leading indent by walking back to the line start of @Test.
        method_start = match.start()
        line_start = source.rfind("\n", 0, method_start) + 1
        method_indent = source[line_start:method_start]
        if any(ch not in " \t" for ch in method_indent):
            method_indent = "    "

        # match.end() - 1 is the position of the opening '{'
        open_brace = match.end() - 1
        depth = 0
        for i, ch in enumerate(source[open_brace:], start=open_brace):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close_brace = i
                    new_source = (
                        source[:open_brace + 1]
                        + "\n" + new_body_content + "\n"
                        + method_indent + source[close_brace:]
                    )
                    return new_source, True
        break
    return source, False


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

        # Keep only the highest _assertN per base test
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

        with open(original_path, "r", encoding="utf-8") as f:
            source = f.read()

        injected = 0
        for base_name, row in test_groups.items():
            prefix = row.get("test_prefix", "")
            assertion = row.get("assert_pred", "")
            if not prefix or not assertion:
                continue

            # The test_prefix from extractor includes the opening "{" line and may include
            # leading whitespace from the original file. Strip just the wrapping braces if
            # present so the body content is clean.
            body_text = prefix.strip()
            if body_text.startswith("{"):
                body_text = body_text[1:]
            if body_text.endswith("}"):
                body_text = body_text[:-1]
            body_text = body_text.strip("\n")

            indent = _detect_body_indent(body_text) or "        "
            assertion_clean = assertion.strip()
            if not assertion_clean.endswith(";") and "assertThrows" not in assertion_clean:
                assertion_clean = assertion_clean.rstrip() + ";"
            indented_assertion = "\n".join(
                indent + line.lstrip() for line in assertion_clean.split("\n")
            )

            new_body = body_text + "\n" + indented_assertion

            source, replaced = _replace_method_body(source, base_name, new_body)
            if replaced:
                injected += 1

        if injected > 0:
            with open(original_path, "w", encoding="utf-8") as f:
                f.write(source)
            modified_files.append(original_path)
            print(f"Injected {injected} test(s) into {original_path}")

    return modified_files

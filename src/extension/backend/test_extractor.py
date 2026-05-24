"""Extract test cases from Python/Java test files into infer_input/{inputs,meta_llm}.csv.

Schema matches what src/solution/run_pipeline.py expects:
  inputs.csv:   focal_method, docstring
  meta_llm.csv: test_name, test_prefix, file_path, GT_output
"""
import ast
import csv
import os
import re
import textwrap
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_csvs(project_path: str, test_cases: list[dict]) -> dict:
    out_dir = Path(project_path) / "infer_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = str(out_dir / "inputs.csv")
    meta_path = str(out_dir / "meta_llm.csv")

    with open(inputs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["focal_method", "docstring"],
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for tc in test_cases:
            writer.writerow({
                "focal_method": tc.get("focal_method", ""),
                "docstring": tc.get("docstring", ""),
            })

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "test_prefix", "file_path", "GT_output"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for tc in test_cases:
            writer.writerow({
                "test_name": tc.get("test_name", ""),
                "test_prefix": tc.get("test_prefix", ""),
                "file_path": tc.get("file_path", ""),
                "GT_output": tc.get("GT_output", ""),
            })

    return {
        "test_count": len(test_cases),
        "inputs_csv": inputs_path,
        "meta_csv": meta_path,
    }


# ---------------------------------------------------------------------------
# Python: focal-method search
# ---------------------------------------------------------------------------

def _find_function_source_python(func_name: str, src_dirs: list[str]) -> Optional[str]:
    pattern = re.compile(
        r"((?:^[ \t]*@\S+\s*\n)*^[ \t]*def\s+" + re.escape(func_name) +
        r"\s*\(.*?)(?=^\S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    code = Path(fpath).read_text(encoding="utf-8", errors="replace")
                    match = pattern.search(code)
                    if match:
                        return match.group(0).rstrip()
                except Exception:
                    continue
    return None


def _last_call_name_from_prefix(prefix: str) -> Optional[str]:
    try:
        tree = ast.parse(textwrap.dedent(prefix))
    except SyntaxError:
        return None

    last_call_name: Optional[str] = None
    trivial = {"len", "range", "str", "int", "float", "list", "dict", "set",
               "tuple", "bool", "print", "type", "isinstance", "hasattr",
               "getattr", "setattr", "enumerate", "zip", "map", "filter",
               "sorted", "reversed", "sum", "min", "max", "abs"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name and name not in trivial and not name.startswith("_"):
                last_call_name = name
    return last_call_name


def _find_python_focal(test_name: str, test_prefix: str, project_path: str) -> str:
    src_dirs = []
    for candidate in ["src", "lib", "."]:
        d = os.path.join(project_path, candidate)
        if os.path.isdir(d):
            src_dirs.append(d)

    stripped = re.sub(r"^test_+", "", test_name)
    stripped = re.sub(r"_assert\d+$", "", stripped)
    if stripped and stripped != test_name:
        src = _find_function_source_python(stripped, src_dirs)
        if src:
            return src

    call_name = _last_call_name_from_prefix(test_prefix)
    if call_name:
        src = _find_function_source_python(call_name, src_dirs)
        if src:
            return src

    return test_prefix


# ---------------------------------------------------------------------------
# Python: test extractor
# ---------------------------------------------------------------------------

def _split_test_into_cases(test_name: str, test_lines: list[str],
                           file_name: str) -> list[dict]:
    test_cases = []
    test_prefix: list[str] = []
    assert_count = 0

    for line in test_lines:
        stripped = line.strip()
        if stripped.startswith("assert ") or stripped.startswith("assert("):
            assert_count += 1
            test_prefix_str = "\n".join(test_prefix)
            gt_output = stripped
            curr_name = test_name if assert_count == 1 else f"{test_name}_assert{assert_count}"
            test_cases.append({
                "test_name": curr_name,
                "test_prefix": test_prefix_str,
                "file_path": file_name,
                "GT_output": gt_output,
            })
            test_prefix.append(line)
        else:
            test_prefix.append(line)

    if assert_count == 0:
        test_cases.append({
            "test_name": test_name,
            "test_prefix": "\n".join(test_prefix),
            "file_path": file_name,
            "GT_output": "",
        })
    return test_cases


def extract_python_tests(project_path: str, progress_callback=None) -> dict:
    project_path = str(project_path)

    test_files: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in {"__pycache__", "node_modules",
                                                   "build", "dist", ".git"}
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, project_path).replace("\\", "/")
                test_files.append((abs_path, rel_path))

    total_files = len(test_files)
    all_test_cases: list[dict] = []

    for file_idx, (abs_path, rel_path) in enumerate(test_files):
        if progress_callback:
            try:
                progress_callback({
                    "type": "extraction",
                    "current": file_idx + 1,
                    "total": total_files,
                    "file": rel_path,
                })
            except Exception:
                pass

        try:
            source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        try:
            tree = ast.parse(source, filename=abs_path)
        except SyntaxError:
            continue

        source_lines = source.splitlines()

        parent_map = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[id(child)] = parent

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue

            parent = parent_map.get(id(node))
            class_name = parent.name if isinstance(parent, ast.ClassDef) else None
            test_name = f"{class_name}.{node.name}" if class_name else node.name

            try:
                body_start = node.body[0].lineno - 1
                body_end = node.end_lineno
                def_lines = source_lines[node.lineno - 1: body_start]
                body_lines = source_lines[body_start:body_end]
            except (AttributeError, IndexError):
                continue

            body_text = "\n".join(def_lines + body_lines)
            full_lines = textwrap.dedent(body_text).splitlines()

            cases = _split_test_into_cases(test_name, full_lines, rel_path)
            all_test_cases.extend(cases)

    for tc in all_test_cases:
        tc["focal_method"] = _find_python_focal(tc["test_name"], tc["test_prefix"], project_path)
        tc["docstring"] = ""

    return _write_csvs(project_path, all_test_cases)


# ---------------------------------------------------------------------------
# Java: focal-method search
# ---------------------------------------------------------------------------

def _find_method_source_java(method_name: str, src_dirs: list[str]) -> Optional[str]:
    sig_pattern = re.compile(
        r"((?:(?:public|private|protected|static|final|synchronized|abstract|"
        r"native|strictfp)\s+)*"
        r"(?:<[^>]+>\s+)?"
        r"[\w\[\]<>,.?]+\s+"
        r"\b" + re.escape(method_name) + r"\s*\([^)]*\)"
        r"(?:\s*throws\s+[\w,\s]+)?\s*\{)"
    )

    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                if not fname.endswith(".java"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    code = Path(fpath).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                match = sig_pattern.search(code)
                if not match:
                    continue

                start = match.end() - 1
                depth = 0
                for i, ch in enumerate(code[start:], start=start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return code[match.start():i + 1]
    return None


def _last_call_name_from_java_prefix(prefix: str) -> Optional[str]:
    trivial = {"assertEquals", "assertNotNull", "assertNull", "assertTrue",
               "assertFalse", "assertThat", "assertSame", "assertNotSame",
               "fail", "verify", "when", "given", "then", "mock", "spy",
               "doReturn", "doNothing", "doThrow"}
    calls = re.findall(r"\b([a-zA-Z_]\w*)\s*\(", prefix)
    for name in reversed(calls):
        if name not in trivial and not name[0].isupper():
            return name
    return None


def _split_java_test_into_cases(test_name: str, test_lines: list[str],
                                file_name: str) -> list[dict]:
    test_cases = []
    test_prefix: list[str] = []
    assert_count = 0

    assert_starts = ("assertEquals", "assertNotNull", "assertNull", "assertTrue",
                     "assertFalse", "assertThat", "assertSame", "assertNotSame",
                     "assertArrayEquals", "assertThrows", "fail(", "assert ")

    for line in test_lines:
        stripped = line.strip()
        if any(stripped.startswith(a) for a in assert_starts):
            assert_count += 1
            test_prefix_str = "\n".join(test_prefix)
            gt_output = stripped
            curr_name = test_name if assert_count == 1 else f"{test_name}_assert{assert_count}"
            test_cases.append({
                "test_name": curr_name,
                "test_prefix": test_prefix_str,
                "file_path": file_name,
                "GT_output": gt_output,
            })
            test_prefix.append(line)
        else:
            test_prefix.append(line)

    if assert_count == 0:
        test_cases.append({
            "test_name": test_name,
            "test_prefix": "\n".join(test_prefix),
            "file_path": file_name,
            "GT_output": "",
        })
    return test_cases


def extract_java_tests(project_path: str, progress_callback=None) -> dict:
    project_path = str(project_path)

    test_method_pattern = re.compile(
        r"@Test[^\n]*\n\s*(?:@[^\n]+\n\s*)*"
        r"(?:public\s+)?(?:void|[\w<>\[\]]+)\s+"
        r"(test\w*|should\w*|\w+Test\w*)\s*\([^)]*\)"
        r"(?:\s*throws\s+[\w,\s]+)?\s*\{",
        re.MULTILINE,
    )

    test_files: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in {"target", "build", ".git",
                                                   "node_modules"}
        ]
        for fname in files:
            if not fname.endswith(".java"):
                continue
            if (fname.endswith("Test.java") or fname.endswith("Tests.java")
                    or fname.startswith("Test")):
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, project_path).replace("\\", "/")
                test_files.append((abs_path, rel_path))

    total_files = len(test_files)
    all_test_cases: list[dict] = []

    src_dirs = []
    for candidate in ["src/main/java", "src", "lib", "."]:
        d = os.path.join(project_path, candidate)
        if os.path.isdir(d):
            src_dirs.append(d)

    for file_idx, (abs_path, rel_path) in enumerate(test_files):
        if progress_callback:
            try:
                progress_callback({
                    "type": "extraction",
                    "current": file_idx + 1,
                    "total": total_files,
                    "file": rel_path,
                })
            except Exception:
                pass

        try:
            source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for match in test_method_pattern.finditer(source):
            test_name = match.group(1)
            method_start = match.end() - 1

            depth = 0
            body_end = method_start
            for i, ch in enumerate(source[method_start:], start=method_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        body_end = i
                        break

            body = source[method_start:body_end + 1]
            body_lines = body.splitlines()

            cases = _split_java_test_into_cases(test_name, body_lines, rel_path)
            all_test_cases.extend(cases)

    for tc in all_test_cases:
        stripped = re.sub(r"^test_*", "", tc["test_name"], flags=re.IGNORECASE)
        stripped = re.sub(r"_assert\d+$", "", stripped)
        focal = None
        if stripped and stripped.lower() != tc["test_name"].lower():
            focal = _find_method_source_java(stripped, src_dirs)
        if not focal:
            call_name = _last_call_name_from_java_prefix(tc["test_prefix"])
            if call_name:
                focal = _find_method_source_java(call_name, src_dirs)
        tc["focal_method"] = focal or tc["test_prefix"]
        tc["docstring"] = ""

    return _write_csvs(project_path, all_test_cases)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_tests(project_path: str, language: str = "python",
                  progress_callback=None) -> dict:
    if language == "python":
        return extract_python_tests(project_path, progress_callback)
    elif language == "java":
        return extract_java_tests(project_path, progress_callback)
    else:
        raise ValueError(f"Unsupported language: {language}")

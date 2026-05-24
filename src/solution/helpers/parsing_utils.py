import re

import pandas as pd


# Tool-arg parsing — recover from malformed LLM tool calls

MALFORMED_ARG_RE = re.compile(
    r"""^\s*class_name\s*=\s*['"]([^'"]*)['"]\s*,\s*"""
    r"""method_name\s*=\s*['"]([^'"]*)['"]"""
    r"""(?:\s*,\s*include_callees\s*=\s*(True|False|true|false))?\s*$"""
)


def normalize_args_from_kwargs(class_name: str, method_name: str, include_callees: bool,
                                extra_kwargs: dict) -> tuple[str, str, bool, bool, bool]:
    combined_candidate = (class_name or method_name or "").strip()
    if combined_candidate:
        m = MALFORMED_ARG_RE.match(combined_candidate)
        if m:
            norm_include = include_callees
            if m.group(3) is not None:
                norm_include = m.group(3).lower() == "true"
            return m.group(1).strip(), m.group(2).strip(), norm_include, True, False
        if "class_name" in combined_candidate and "method_name" in combined_candidate:
            return "", "", include_callees, False, True

    if class_name or method_name or not extra_kwargs:
        return class_name, method_name, include_callees, False, False

    if len(extra_kwargs) != 1:
        return class_name, method_name, include_callees, False, False

    malformed_key, malformed_val = next(iter(extra_kwargs.items()))
    if malformed_val not in ("", None):
        return class_name, method_name, include_callees, False, False

    m = MALFORMED_ARG_RE.match(str(malformed_key))
    if not m:
        return class_name, method_name, include_callees, False, True

    norm_class = m.group(1).strip()
    norm_method = m.group(2).strip()
    norm_include = include_callees
    if m.group(3) is not None:
        norm_include = m.group(3).lower() == "true"
    return norm_class, norm_method, norm_include, True, False


# Test metadata extraction

def extract_focal_class(test_name, language='java'):
    # Parse Java test name into class-under-test name.
    # e.g. 'org...ContentWriteProgress_ESTest::test1' -> 'ContentWriteProgress'
    # Nested class: 'Outer_Inner_ESTest' -> 'Outer$Inner'
    if language != 'java':
        return ''
    if '::' not in test_name:
        return ''
    cls = test_name.split('::')[0].rsplit('.', 1)[-1]
    for suffix in ('_ESTest_scaffolding', '_ESTest'):
        if cls.endswith(suffix):
            cls = cls[:-len(suffix)]
            break
    if '_' in cls and '$' not in cls:
        parts = cls.split('_')
        if len(parts) >= 2 and all(parts):
            cls = parts[0] + '$' + '_'.join(parts[1:])
    return cls


def extract_return_type(code):
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    code = str(code).strip()
    if not code or code.lower() == 'nan':
        return None
    match = re.search(
        r'(?:public|private|protected|static|\s)+\s*(?:<[^>]+>\s*)?'
        r'((?:\w+\.)*\w+(?:<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>)?(?:\[\])*)\s+\w+\s*\(',
        code,
    )
    return match.group(1) if match else None

import logging
import os
from typing import Optional

from tree_sitter import Language, Parser, Query, QueryCursor

from lang_config import LANG_CONFIGS

from .queries import LANGUAGE_QUERIES
from .tree_helpers import (
    load_language, detect_language,
    ExtractedClass, ExtractedMethod, ExtractedCall, ExtractedField,
    ExtractedHeritage, FileExtractionResult,
    is_test_file, is_test_class,
    find_enclosing_node, get_node_name,
    extract_return_type_java, extract_parameters, extract_field_modifier,
    CLASS_NODE_TYPES, METHOD_NODE_TYPES, EXCLUDE_DIRS,
)

logger = logging.getLogger(__name__)


class MultiLanguageParser:
    # Tree-sitter parser; caches loaded languages and compiled queries per instance.

    def __init__(self):
        self._parser = Parser()
        self._languages: dict[str, Language] = {}
        self._queries: dict[str, object] = {}

    def ensure_language(self, lang: str) -> bool:
        # Load and cache language grammar. Returns True if available.
        if lang in self._languages:
            return True
        ts_lang = load_language(lang)
        if ts_lang is None:
            return False
        self._languages[lang] = ts_lang
        return True

    def parse_file(self, file_path: str, source_code: str,
                   language: Optional[str] = None) -> Optional[FileExtractionResult]:
        # Parse a single source file and extract classes, methods, calls,
        # fields, and heritage relationships. Returns None if language unsupported.
        lang = language or detect_language(file_path)
        if not lang or lang not in LANGUAGE_QUERIES:
            return None

        if not self.ensure_language(lang):
            return None

        ts_lang = self._languages[lang]
        self._parser.language = ts_lang

        try:
            tree = self._parser.parse(bytes(source_code, "utf8"))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", file_path, e)
            return None

        result = FileExtractionResult()
        query_string = LANGUAGE_QUERIES[lang]

        try:
            query = Query(ts_lang, query_string)
            qc = QueryCursor(query)
            matches = qc.matches(tree.root_node)
        except Exception as e:
            logger.warning("Query error for %s: %s", file_path, e)
            return None

        class_node_types = CLASS_NODE_TYPES.get(lang, set())
        method_node_types = METHOD_NODE_TYPES.get(lang, set())

        for pattern_idx, captures in matches:
            # Helper to get first node from a capture
            def _get(name):
                nodes = captures.get(name)
                return nodes[0] if nodes else None

            # Class definitions
            if "class.def" in captures and "class.name" in captures:
                class_node = _get("class.def")
                class_name_node = _get("class.name")
                if class_node and class_name_node:
                    class_name = class_name_node.text.decode("utf8")
                    if not is_test_class(class_name):
                        result.classes.append(ExtractedClass(
                            name=class_name,
                            file_path=file_path,
                            start_line=class_node.start_point[0] + 1,
                            end_line=class_node.end_point[0] + 1,
                        ))

            # Method definitions
            elif "method.def" in captures and "method.name" in captures:
                method_node = _get("method.def")
                method_name_node = _get("method.name")
                if not method_node or not method_name_node:
                    continue
                method_name = method_name_node.text.decode("utf8")

                # Find enclosing class
                enclosing = find_enclosing_node(method_node, class_node_types)
                enclosing_name = get_node_name(enclosing, lang) if enclosing else ""

                # Skip test classes
                if enclosing_name and is_test_class(enclosing_name):
                    continue

                body_text = method_node.text.decode("utf8")
                return_type = extract_return_type_java(method_node) if lang == "java" else ""
                params = extract_parameters(method_node)

                result.methods.append(ExtractedMethod(
                    name=method_name,
                    class_name=enclosing_name,
                    file_path=file_path,
                    body=body_text,
                    start_line=method_node.start_point[0] + 1,
                    end_line=method_node.end_point[0] + 1,
                    return_type=return_type,
                    parameters=params,
                ))

            # Call expressions
            elif "call" in captures and "call.name" in captures:
                call_node = _get("call")
                callee_node = _get("call.name")
                if not call_node or not callee_node:
                    continue
                callee_name = callee_node.text.decode("utf8")

                # Find enclosing method and class
                enclosing_method = find_enclosing_node(call_node, method_node_types)
                enclosing_class = find_enclosing_node(call_node, class_node_types)

                caller_method_name = get_node_name(enclosing_method, lang) if enclosing_method else ""
                caller_class_name = get_node_name(enclosing_class, lang) if enclosing_class else ""

                # Skip module-level calls without enclosing method.
                if not caller_method_name:
                    continue

                result.calls.append(ExtractedCall(
                    caller_method=caller_method_name,
                    caller_class=caller_class_name,
                    callee_name=callee_name,
                    file_path=file_path,
                    line=call_node.start_point[0] + 1,
                ))

            # Heritage — extends
            elif "heritage" in captures and "heritage.class" in captures and "heritage.extends" in captures:
                cn = _get("heritage.class")
                en = _get("heritage.extends")
                if cn and en:
                    result.heritage.append(ExtractedHeritage(
                        class_name=cn.text.decode("utf8"),
                        extends=en.text.decode("utf8"),
                    ))

            # Heritage — implements
            elif "heritage.impl" in captures and "heritage.class" in captures and "heritage.implements" in captures:
                cn = _get("heritage.class")
                im = _get("heritage.implements")
                if cn and im:
                    result.heritage.append(ExtractedHeritage(
                        class_name=cn.text.decode("utf8"),
                        implements=im.text.decode("utf8"),
                    ))

            # Field declarations
            elif "field.def" in captures and "field.name" in captures and "field.type" in captures:
                field_node = _get("field.def")
                fn = _get("field.name")
                ft = _get("field.type")
                if not field_node or not fn or not ft:
                    continue

                enclosing = find_enclosing_node(field_node, class_node_types)
                enclosing_name = get_node_name(enclosing, lang) if enclosing else ""

                modifier = extract_field_modifier(field_node) if lang == "java" else "public"

                result.fields.append(ExtractedField(
                    name=fn.text.decode("utf8"),
                    field_type=ft.text.decode("utf8"),
                    class_name=enclosing_name,
                    file_path=file_path,
                    modifier=modifier,
                ))

        return result

    def parse_project(self, project_path: str, language: str,
                      file_extensions: Optional[list[str]] = None,
                      on_progress=None) -> list[FileExtractionResult]:
        # Parse all source files in a project directory.
        # on_progress(current, total, file_path) called per file if provided.
        if file_extensions is None:
            cfg = LANG_CONFIGS.get(language, {})
            file_extensions = cfg.get("file_extensions", [f".{language}"])

        # Collect all matching files
        all_files = []
        for root, dirs, files in os.walk(project_path):
            # Skip hidden dirs and common non-source dirs
            dirs[:] = [d for d in dirs if not d.startswith('.')
                       and d not in EXCLUDE_DIRS]
            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext.lower() in file_extensions:
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, project_path).replace("\\", "/")
                    if not is_test_file(rel_path):
                        all_files.append((fpath, rel_path))

        results = []
        total = len(all_files)
        for i, (fpath, rel_path) in enumerate(all_files):
            if on_progress:
                on_progress(i + 1, total, rel_path)

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except Exception as e:
                logger.warning("Could not read %s: %s", fpath, e)
                continue

            extraction = self.parse_file(rel_path, source, language=language)
            if extraction:
                results.append(extraction)

        return results

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from tree_sitter import Language, Parser, Node, Query, QueryCursor

from .queries import LANGUAGE_QUERIES

logger = logging.getLogger(__name__)

# Language grammar loaders
# Each tree-sitter-<lang> pip package exposes a language() function.

_GRAMMAR_LOADERS = {}


def _register_grammar(lang_key: str, module_name: str):
    _GRAMMAR_LOADERS[lang_key] = module_name

# load for 3 languages
_register_grammar("java", "tree_sitter_java")
_register_grammar("python", "tree_sitter_python")
_register_grammar("javascript", "tree_sitter_javascript")


def _load_language(lang_key: str):
    """Load a tree-sitter Language object, returns None if unavailable."""
    module_name = _GRAMMAR_LOADERS.get(lang_key)
    if not module_name:
        return None
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return Language(mod.language())
    except (ImportError, Exception) as e:
        logger.warning("Could not load tree-sitter grammar for %s: %s", lang_key, e)
        return None


# File extension to language mapping

_EXT_TO_LANG = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
}


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension."""
    _, ext = os.path.splitext(file_path)
    return _EXT_TO_LANG.get(ext.lower())


# Extracted data classes

@dataclass
class ExtractedClass:
    name: str
    file_path: str
    start_line: int
    end_line: int
    extends: list[str] = field(default_factory=list)
    implements: list[str] = field(default_factory=list)


@dataclass
class ExtractedMethod:
    name: str
    class_name: str  # enclosing class, or "" for top-level functions
    file_path: str
    body: str
    start_line: int
    end_line: int
    return_type: str = ""
    parameters: str = ""  # raw parameter text


@dataclass
class ExtractedCall:
    """A call relationship: caller_method calls callee_name."""
    caller_method: str  # name of the enclosing method
    caller_class: str  # name of the enclosing class
    callee_name: str  # name of the called method/constructor
    file_path: str
    line: int


@dataclass
class ExtractedField:
    name: str
    field_type: str
    class_name: str
    file_path: str
    modifier: str = "package"  # public/private/protected/package


@dataclass
class ExtractedHeritage:
    class_name: str
    extends: str = ""
    implements: str = ""


@dataclass
class FileExtractionResult:
    """All data extracted from a single source file."""
    classes: list[ExtractedClass] = field(default_factory=list)
    methods: list[ExtractedMethod] = field(default_factory=list)
    calls: list[ExtractedCall] = field(default_factory=list)
    fields: list[ExtractedField] = field(default_factory=list)
    heritage: list[ExtractedHeritage] = field(default_factory=list)


# Filtering helpers

def _is_test_file(fpath: str) -> bool:
    lower = fpath.lower()
    return 'evosuite' in lower or '/test/' in lower


def _is_test_class(name: str) -> bool:
    return (name.endswith('_ESTest') or name.endswith('_ESTest_scaffolding')
            or name.endswith('Test') or name.startswith('Test')
            or name.endswith('_test') or name.startswith('test_'))


# CST helper: find enclosing class/method

def _find_enclosing_node(node: Node, type_names: set[str]) -> Optional[Node]:
    """Walk up the AST to find the nearest enclosing node of given type(s)."""
    current = node.parent
    while current:
        if current.type in type_names:
            return current
        current = current.parent
    return None


# Class node types per language
_CLASS_NODE_TYPES = {
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
}

# Method node types per language
_METHOD_NODE_TYPES = {
    "java": {"method_declaration", "constructor_declaration"},
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition"},
}

_EXCLUDE_DIRS = {
    "node_modules", "__pycache__", "build", "target", ".git",
    ".idea", ".vscode", ".gradle", "dist", "out", "coverage",
}


def _get_node_name(node: Node, lang: str) -> str:
    """Extract the 'name' child from a class/method AST node."""
    # For most nodes, the name is a named child called 'name'
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf8")
    return ""


def _extract_return_type_java(method_node: Node) -> str:
    """Extract return type from a Java method_declaration node."""
    if method_node.type == "constructor_declaration":
        return ""
    type_node = method_node.child_by_field_name("type")
    if type_node:
        return type_node.text.decode("utf8")
    return ""


def _extract_parameters(method_node: Node) -> str:
    """Extract raw parameter text from a method node."""
    params_node = method_node.child_by_field_name("parameters")
    if params_node:
        text = params_node.text.decode("utf8")
        # Strip outer parens
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return text.strip()
    return ""


def _extract_field_modifier(field_node: Node) -> str:
    """Extract visibility modifier from a Java field declaration."""
    for child in field_node.children:
        if child.type == "modifiers":
            text = child.text.decode("utf8")
            if "public" in text:
                return "public"
            if "private" in text:
                return "private"
            if "protected" in text:
                return "protected"
    return "package"


# Main parser class

class MultiLanguageParser:
    """Parse source files using tree-sitter and extract structured code data."""

    def __init__(self):
        self._parser = Parser()
        self._languages: dict[str, Language] = {}
        self._queries: dict[str, object] = {}

    def _ensure_language(self, lang: str) -> bool:
        """Load and cache language grammar. Returns True if available."""
        if lang in self._languages:
            return True
        ts_lang = _load_language(lang)
        if ts_lang is None:
            return False
        self._languages[lang] = ts_lang
        return True

    def parse_file(self, file_path: str, source_code: str,
                   language: Optional[str] = None) -> Optional[FileExtractionResult]:
        """Parse a single source file and extract all code elements.

        Args:
            file_path: Path to the source file (for metadata).
            source_code: Raw source code string.
            language: Explicitly specify language, or auto-detect from extension.

        Returns:
            FileExtractionResult or None if language not supported.
        """
        lang = language or detect_language(file_path)
        if not lang or lang not in LANGUAGE_QUERIES:
            return None

        if not self._ensure_language(lang):
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

        class_node_types = _CLASS_NODE_TYPES.get(lang, set())
        method_node_types = _METHOD_NODE_TYPES.get(lang, set())

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
                    if not _is_test_class(class_name):
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
                enclosing = _find_enclosing_node(method_node, class_node_types)
                enclosing_name = _get_node_name(enclosing, lang) if enclosing else ""

                # Skip test classes
                if enclosing_name and _is_test_class(enclosing_name):
                    continue

                body_text = method_node.text.decode("utf8")
                return_type = _extract_return_type_java(method_node) if lang == "java" else ""
                params = _extract_parameters(method_node)

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
                enclosing_method = _find_enclosing_node(call_node, method_node_types)
                enclosing_class = _find_enclosing_node(call_node, class_node_types)

                caller_method_name = _get_node_name(enclosing_method, lang) if enclosing_method else ""
                caller_class_name = _get_node_name(enclosing_class, lang) if enclosing_class else ""

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

                enclosing = _find_enclosing_node(field_node, class_node_types)
                enclosing_name = _get_node_name(enclosing, lang) if enclosing else ""

                modifier = _extract_field_modifier(field_node) if lang == "java" else "public"

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
        """Parse all source files in a project directory.

        Args:
            project_path: Root directory of the project.
            language: Language to parse (e.g., "java").
            file_extensions: Only parse files with these extensions.
                             Default: auto from language config.
            on_progress: Callback(current, total, file_path).

        Returns:
            List of FileExtractionResult for each parsed file.
        """
        from lang_config import LANG_CONFIGS

        if file_extensions is None:
            cfg = LANG_CONFIGS.get(language, {})
            file_extensions = cfg.get("file_extensions", [f".{language}"])

        # Collect all matching files
        all_files = []
        for root, dirs, files in os.walk(project_path):
            # Skip hidden dirs and common non-source dirs
            dirs[:] = [d for d in dirs if not d.startswith('.')
                       and d not in _EXCLUDE_DIRS]
            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext.lower() in file_extensions:
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, project_path).replace("\\", "/")
                    if not _is_test_file(rel_path):
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

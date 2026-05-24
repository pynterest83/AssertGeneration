import logging
from dataclasses import dataclass, field
from typing import Optional

from tree_sitter import Language, Node

logger = logging.getLogger(__name__)


# Grammar loaders — each tree-sitter-<lang> pip package exposes a language() function.

GRAMMAR_LOADERS = {}


def register_grammar(lang_key: str, module_name: str):
    GRAMMAR_LOADERS[lang_key] = module_name


# load for 3 languages
register_grammar("java", "tree_sitter_java")
register_grammar("python", "tree_sitter_python")
register_grammar("javascript", "tree_sitter_javascript")


def load_language(lang_key: str):
    # Load a tree-sitter Language object; returns None if unavailable.
    module_name = GRAMMAR_LOADERS.get(lang_key)
    if not module_name:
        return None
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return Language(mod.language())
    except (ImportError, Exception) as e:
        logger.warning("Could not load tree-sitter grammar for %s: %s", lang_key, e)
        return None


# File extension → language mapping

EXT_TO_LANG = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
}


def detect_language(file_path: str) -> Optional[str]:
    import os
    _, ext = os.path.splitext(file_path)
    return EXT_TO_LANG.get(ext.lower())


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
    # A call relationship: caller_method calls callee_name.
    caller_method: str
    caller_class: str
    callee_name: str
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
    # All data extracted from a single source file.
    classes: list[ExtractedClass] = field(default_factory=list)
    methods: list[ExtractedMethod] = field(default_factory=list)
    calls: list[ExtractedCall] = field(default_factory=list)
    fields: list[ExtractedField] = field(default_factory=list)
    heritage: list[ExtractedHeritage] = field(default_factory=list)


# Filtering helpers

def is_test_file(fpath: str) -> bool:
    lower = fpath.lower()
    return 'evosuite' in lower or '/test/' in lower


def is_test_class(name: str) -> bool:
    return (name.endswith('_ESTest') or name.endswith('_ESTest_scaffolding')
            or name.endswith('Test') or name.startswith('Test')
            or name.endswith('_test') or name.startswith('test_'))


# CST traversal: find enclosing class/method

def find_enclosing_node(node: Node, type_names: set[str]) -> Optional[Node]:
    # Walk up the parents to find the nearest enclosing node of given type(s).
    current = node.parent
    while current:
        if current.type in type_names:
            return current
        current = current.parent
    return None


# Per-language node-type tables

CLASS_NODE_TYPES = {
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
}

METHOD_NODE_TYPES = {
    "java": {"method_declaration", "constructor_declaration"},
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition"},
}

EXCLUDE_DIRS = {
    "node_modules", "__pycache__", "build", "target", ".git",
    ".idea", ".vscode", ".gradle", "dist", "out", "coverage",
}


def get_node_name(node: Node, lang: str) -> str:
    # For most nodes, the name is a named child called 'name'.
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf8")
    return ""


def extract_return_type_java(method_node: Node) -> str:
    # Extract return type from a Java method_declaration node.
    if method_node.type == "constructor_declaration":
        return ""
    type_node = method_node.child_by_field_name("type")
    if type_node:
        return type_node.text.decode("utf8")
    return ""


def extract_parameters(method_node: Node) -> str:
    # Extract raw parameter text from a method node.
    params_node = method_node.child_by_field_name("parameters")
    if params_node:
        text = params_node.text.decode("utf8")
        # Strip outer parens
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return text.strip()
    return ""


def extract_field_modifier(field_node: Node) -> str:
    # Extract visibility modifier from a Java field declaration.
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

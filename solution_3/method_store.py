import os
import javalang
from collections import defaultdict
from tqdm import tqdm

def _is_test_file(fpath):
    lower = fpath.lower()
    return 'evosuite' in lower or '/test/' in lower or '\\test\\' in lower


def _is_test_class(name):
    return (name.endswith('_ESTest') or name.endswith('_ESTest_scaffolding')
            or name.endswith('Test') or name.startswith('Test')
            or name.endswith('_test') or name.startswith('test_'))


class MethodInfo:
    __slots__ = ['name', 'class_name', 'return_type', 'parameters', 'body']

    def __init__(self, name, class_name, return_type, parameters, body):
        self.name = name
        self.class_name = class_name
        self.return_type = return_type
        self.parameters = parameters
        self.body = body

    def signature(self):
        params = ', '.join(f"{t} {n}" for t, n in self.parameters)
        return f"{self.class_name}.{self.name}({params}) -> {self.return_type}"

    def format(self):
        return f"// {self.signature()}\n{self.body}"


class ClassInfo:
    __slots__ = ['name', 'extends', 'implements', 'fields', 'field_modifiers']

    def __init__(self, name, extends=None, implements=None):
        self.name = name
        self.extends = extends
        self.implements = implements or []
        self.fields = {}
        self.field_modifiers = {}  # {field_name: 'public'/'private'/'protected'/'package'}


class MethodStore:
    """AST-based index of a Java project.

    Parses all .java files with javalang, indexes:
    - Class metadata (extends, implements, fields with types)
    - All method/constructor declarations (including nested/anonymous classes)
    """

    def __init__(self, project_path):
        self.classes = {}
        self.by_class = defaultdict(list)
        self.by_name = defaultdict(list)
        self.by_class_method = defaultdict(list)
        self.field_types = {}
        self.subclasses = defaultdict(list)

        self._parse_project(project_path)

    def _parse_project(self, project_path):
        java_files = []
        for root, _, files in os.walk(project_path):
            for f in files:
                if f.endswith('.java'):
                    java_files.append(os.path.join(root, f))

        for fpath in tqdm(java_files, desc="Building AST index", leave=False):
            if _is_test_file(fpath):
                continue
            self._parse_file(fpath)

    def _parse_file(self, fpath):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                source = f.read().replace('\r\n', '\n')
            tree = javalang.parse.parse(source)
        except Exception:
            return

        self._collect_class_metadata(tree)
        self._collect_methods(tree, source)

    def _collect_class_metadata(self, tree):
        """Extract class-level info: extends, implements, field types."""
        class_types = (
            javalang.tree.ClassDeclaration,
            javalang.tree.InterfaceDeclaration,
            javalang.tree.EnumDeclaration,
        )
        for _, node in tree:
            if not isinstance(node, class_types):
                continue
            cname = node.name
            if _is_test_class(cname):
                continue

            extends = None
            if hasattr(node, 'extends') and node.extends:
                ext = node.extends
                if isinstance(ext, list):
                    extends = [e.name for e in ext if hasattr(e, 'name')]
                elif hasattr(ext, 'name'):
                    extends = [ext.name]

            implements = []
            if hasattr(node, 'implements') and node.implements:
                for impl in node.implements:
                    if hasattr(impl, 'name'):
                        implements.append(impl.name)

            ci = self.classes.get(cname, ClassInfo(cname))
            ci.extends = extends
            ci.implements = implements

            if hasattr(node, 'body') and node.body:
                for member in node.body:
                    if isinstance(member, javalang.tree.FieldDeclaration):
                        ftype = _get_type_name(member.type)
                        mods = member.modifiers or set()
                        vis = ('private' if 'private' in mods else
                               'public' if 'public' in mods else
                               'protected' if 'protected' in mods else
                               'package')
                        for decl in member.declarators:
                            ci.fields[decl.name] = ftype
                            ci.field_modifiers[decl.name] = vis
                            self.field_types[(cname, decl.name)] = ftype

            self.classes[cname] = ci

            if ci.extends:
                for parent in ci.extends:
                    self.subclasses[parent].append(cname)
            for parent in ci.implements:
                self.subclasses[parent].append(cname)

    def _collect_methods(self, tree, source):
        """Walk ALL MethodDeclaration/ConstructorDeclaration nodes in the tree.
        Attributes each to its nearest named parent class (handles anonymous classes)."""
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            class_name = _find_parent_class(path)
            if not class_name or _is_test_class(class_name):
                continue
            mi = self._make_method_info(node, class_name, source)
            if mi:
                self._index_method(mi)

        for path, node in tree.filter(javalang.tree.ConstructorDeclaration):
            class_name = _find_parent_class(path)
            if not class_name or _is_test_class(class_name):
                continue
            mi = self._make_method_info(node, class_name, source, is_constructor=True)
            if mi:
                self._index_method(mi)

    def _make_method_info(self, node, class_name, source, is_constructor=False):
        if is_constructor:
            ret = class_name
        elif hasattr(node, 'return_type') and node.return_type:
            ret = _get_type_name(node.return_type)
        else:
            ret = 'void'

        params = []
        if node.parameters:
            for p in node.parameters:
                ptype = _get_type_name(p.type) if p.type else 'Object'
                params.append((ptype, p.name))

        body = _get_body(source, node)
        return MethodInfo(node.name, class_name, ret, params, body)

    def _index_method(self, mi):
        self.by_class[mi.class_name].append(mi)
        self.by_name[mi.name].append(mi)
        self.by_class_method[(mi.class_name, mi.name)].append(mi)

    def search(self, class_name=None, method_name=None, max_results=10):
        if class_name and method_name:
            results = self._resolve_method(class_name, method_name)
        elif class_name:
            results = list(self.by_class.get(class_name, []))
        elif method_name:
            results = list(self.by_name.get(method_name, []))
        else:
            return []
        return results[:max_results]

    def _resolve_method(self, class_name, method_name):
        """Resolve a method through the inheritance hierarchy.

        1. Direct match (class_name, method_name)
        2. Walk UP extends/implements (find declaration in a parent within the project)
        3. Walk DOWN subclasses of the *original* class (find concrete implementation)
        """
        hit = self.by_class_method.get((class_name, method_name))
        if hit:
            return list(hit)

        found = self._walk_up(class_name, method_name, set())
        if found:
            return found

        return self._walk_down(class_name, method_name, set())

    def _walk_up(self, class_name, method_name, visited):
        if class_name in visited:
            return []
        visited.add(class_name)
        ci = self.classes.get(class_name)
        if not ci:
            return []
        for parent in (ci.extends or []) + ci.implements:
            hit = self.by_class_method.get((parent, method_name))
            if hit:
                return list(hit)
            found = self._walk_up(parent, method_name, visited)
            if found:
                return found
        return []

    def _walk_down(self, class_name, method_name, visited):
        if class_name in visited:
            return []
        visited.add(class_name)
        for child in self.subclasses.get(class_name, []):
            hit = self.by_class_method.get((child, method_name))
            if hit:
                return list(hit)
            found = self._walk_down(child, method_name, visited)
            if found:
                return found
        return []

    def resolve_field_type(self, class_name, field_name):
        return self.field_types.get((class_name, field_name))

    def get_class_info(self, class_name):
        return self.classes.get(class_name)


def _find_parent_class(path):
    """Walk up the AST path to find the nearest named class."""
    for parent in reversed(path):
        if isinstance(parent, (javalang.tree.ClassDeclaration,
                               javalang.tree.InterfaceDeclaration,
                               javalang.tree.EnumDeclaration)):
            return parent.name
    return None


def _get_type_name(type_node):
    if type_node is None:
        return 'Object'
    if hasattr(type_node, 'name'):
        name = type_node.name
        if hasattr(type_node, 'arguments') and type_node.arguments:
            args = ', '.join(
                _get_type_name(a.type) if hasattr(a, 'type') and a.type is not None else '?'
                for a in type_node.arguments if a is not None
            )
            if args:
                name = f"{name}<{args}>"
        if hasattr(type_node, 'dimensions') and type_node.dimensions:
            name += '[]' * len(type_node.dimensions)
        return name
    return str(type_node)


def _get_body(source, node):
    if not node.position:
        return ''

    lines = source.split('\n')
    start_line = node.position.line - 1
    char_pos = sum(len(lines[i]) + 1 for i in range(start_line))

    if getattr(node, 'modifiers', None) and 'abstract' in node.modifiers:
        end = start_line
        while end < len(lines) and ';' not in lines[end]:
            end += 1
        return '\n'.join(lines[start_line:end + 1]).strip()

    semi_pos = source.find(';', char_pos)
    brace_pos = source.find('{', char_pos)

    if brace_pos == -1:
        return ''

    if semi_pos != -1 and semi_pos < brace_pos:
        return '\n'.join(lines[start_line:]).split(';')[0].strip() + ';'

    depth = 0
    for i, ch in enumerate(source[brace_pos:], start=brace_pos):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return source[char_pos:i + 1].strip()
    return ''

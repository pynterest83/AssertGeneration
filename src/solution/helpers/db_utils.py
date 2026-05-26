import re


# Data classes

class MethodInfo:
    __slots__ = ['name', 'class_name', 'return_type', 'parameters', 'body']

    def __init__(self, name, class_name, return_type, parameters, body):
        self.name = name
        self.class_name = class_name
        self.return_type = return_type
        self.parameters = parameters  # raw string "Type1 name1, Type2 name2"
        self.body = body

    def signature(self):
        return f"{self.class_name}.{self.name}({self.parameters}) -> {self.return_type}"

    def format(self):
        return f"// {self.signature()}\n{self.body}"


class ClassInfo:
    __slots__ = ['name', 'extends', 'implements', 'fields', 'field_modifiers']

    def __init__(self, name, extends=None, implements=None):
        self.name = name
        self.extends = extends or []
        self.implements = implements or []
        self.fields = {}            # {field_name: field_type}
        self.field_modifiers = {}   # {field_name: "public"/"private"/...}

#   class Dog extends Animal
# Fields:
#   private int age;
#  ---
#  // Dog.bark() -> void
#  // Dog.getAge() -> int
def format_class_header(ci: ClassInfo) -> str:
    # Render ClassInfo as Java-like header + fields summary (for LLM consumption).
    header_parts = [f"class {ci.name}"]
    if ci.extends:
        header_parts.append(f"extends {', '.join(ci.extends)}")
    if ci.implements:
        header_parts.append(f"implements {', '.join(ci.implements)}")
    lines = [' '.join(header_parts)]
    if ci.fields:
        fields_str = '\n'.join(
            f"  {ci.field_modifiers.get(fname, 'package')} {ftype} {fname};"
            for fname, ftype in ci.fields.items()
        )
        lines.append(f"Fields:\n{fields_str}")
    return '\n'.join(lines)


# ID helpers

def generate_id(label: str, qualname: str) -> str:
    return f"{label}:{qualname}"


def param_key(params: str) -> str:
    return re.sub(r'\s+', '', params).replace(',', '_') if params else ''


# DB Schema

SCHEMA_SQL = [
    # Node tables
    """CREATE NODE TABLE IF NOT EXISTS Class(
        id STRING,
        name STRING,
        filePath STRING,
        startLine INT64,
        endLine INT64,
        PRIMARY KEY(id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Method(
        id STRING,
        name STRING,
        className STRING,
        filePath STRING,
        body STRING,
        returnType STRING,
        parameters STRING,
        startLine INT64,
        endLine INT64,
        PRIMARY KEY(id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Field(
        id STRING,
        name STRING,
        fieldType STRING,
        className STRING,
        modifier STRING,
        PRIMARY KEY(id)
    )""",
    # Relationship tables
    "CREATE REL TABLE IF NOT EXISTS HAS_METHOD(FROM Class TO Method)",
    "CREATE REL TABLE IF NOT EXISTS CALLS(FROM Method TO Method)",
    "CREATE REL TABLE IF NOT EXISTS EXTENDS(FROM Class TO Class)",
    "CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(FROM Class TO Class)",
    "CREATE REL TABLE IF NOT EXISTS HAS_FIELD(FROM Class TO Field)",
]


class Queries:
    # Cypher query constants. Access via Queries.INSERT_CLASS, etc.
    # Grouped by purpose for readability — no functional difference.

    # Sentinel + counts
    INSERT_SENTINEL_METHOD = (
        "CREATE (m:Method {id: 'sentinel:__indexed_complete__', "
        "name: '__indexed_complete__', className: '__sentinel__', "
        "filePath: '', body: '', returnType: '', parameters: '', "
        "startLine: 0, endLine: 0})"
    )
    COUNT_METHODS_EXCLUDING_SENTINEL = (
        "MATCH (m:Method) WHERE m.className <> '__sentinel__' RETURN count(m)"
    )
    COUNT_CLASSES = "MATCH (c:Class) RETURN count(c)"

    # Insert (nodes)
    INSERT_CLASS = (
        "CREATE (c:Class {id: $id, name: $name, filePath: $fp, "
        "startLine: $sl, endLine: $el})"
    )
    INSERT_METHOD = (
        "CREATE (m:Method {id: $id, name: $name, className: $cn, "
        "filePath: $fp, body: $body, returnType: $rt, parameters: $params, "
        "startLine: $sl, endLine: $el})"
    )
    INSERT_FIELD = (
        "CREATE (f:Field {id: $id, name: $name, fieldType: $ft, "
        "className: $cn, modifier: $mod})"
    )

    # Insert (edges)
    INSERT_HAS_METHOD_EDGE = (
        "MATCH (c:Class {id: $cid}), (m:Method {id: $mid}) "
        "CREATE (c)-[:HAS_METHOD]->(m)"
    )
    INSERT_CALLS_EDGE = (
        "MATCH (a:Method {id: $aid}), (b:Method {id: $bid}) "
        "CREATE (a)-[:CALLS]->(b)"
    )
    INSERT_EXTENDS_EDGE = (
        "MATCH (a:Class {id: $aid}), (b:Class {id: $bid}) "
        "CREATE (a)-[:EXTENDS]->(b)"
    )
    INSERT_IMPLEMENTS_EDGE = (
        "MATCH (a:Class {id: $aid}), (b:Class {id: $bid}) "
        "CREATE (a)-[:IMPLEMENTS]->(b)"
    )
    INSERT_HAS_FIELD_EDGE = (
        "MATCH (c:Class {id: $cid}), (f:Field {id: $fid}) "
        "CREATE (c)-[:HAS_FIELD]->(f)"
    )

    # Lookup (during CALLS edge construction)
    LOOKUP_CALLER_BY_FILEPATH = (
        "MATCH (m:Method {className: $cn, name: $mn}) "
        "WHERE m.filePath = $fp RETURN m.id LIMIT 1"
    )
    LOOKUP_CALLEE_IN_SAME_CLASS = (
        "MATCH (m:Method {name: $name, className: $cn}) RETURN m.id LIMIT 1"
    )
    LOOKUP_CALLEE_ANY_CLASS = (
        "MATCH (m:Method {name: $name}) "
        "WHERE m.className <> '__sentinel__' RETURN m.id LIMIT 1"
    )
    CHECK_CALLS_EDGE_EXISTS = (
        "MATCH (a:Method {id: $aid})-[:CALLS]->(b:Method {id: $bid}) "
        "RETURN count(*)"
    )

    # Class cache loading
    LOAD_ALL_CLASSES = "MATCH (c:Class) RETURN c.name"
    LOAD_EXTENDS_EDGES = (
        "MATCH (a:Class)-[:EXTENDS]->(b:Class) RETURN a.name, b.name"
    )
    LOAD_IMPLEMENTS_EDGES = (
        "MATCH (a:Class)-[:IMPLEMENTS]->(b:Class) RETURN a.name, b.name"
    )
    LOAD_HAS_FIELD_EDGES = (
        "MATCH (c:Class)-[:HAS_FIELD]->(f:Field) "
        "RETURN c.name, f.name, f.fieldType, f.modifier"
    )

    # Search
    SEARCH_BY_CLASS = (
        "MATCH (m:Method {className: $cn}) "
        "WHERE m.className <> '__sentinel__' "
        "RETURN m.name, m.className, m.returnType, m.parameters, m.body "
        "LIMIT $limit"
    )
    SEARCH_BY_NAME_STANDALONE = (
        "MATCH (m:Method {name: $name}) "
        "WHERE m.className = '' "
        "RETURN m.name, m.className, m.returnType, m.parameters, m.body "
        "LIMIT $limit"
    )
    SEARCH_BY_NAME = (
        "MATCH (m:Method {name: $name}) "
        "WHERE m.className <> '__sentinel__' "
        "RETURN m.name, m.className, m.returnType, m.parameters, m.body "
        "LIMIT $limit"
    )
    SEARCH_CLASS_METHOD = (
        "MATCH (m:Method {className: $cn, name: $name}) "
        "RETURN m.name, m.className, m.returnType, m.parameters, m.body "
        "LIMIT $limit"
    )
    LOOKUP_METHOD_IDS_BY_CLASS_NAME = (
        "MATCH (m:Method {className: $cn, name: $mn}) RETURN m.id"
    )

    @staticmethod
    def search_callees(depth: int) -> str:
        # f-string template — depth must be interpolated into Cypher [:CALLS*1..N].
        return (
            f"MATCH (a:Method {{id: $aid}})-[:CALLS*1..{depth}]->(b:Method) "
            "WHERE b.className <> '__sentinel__' "
            "RETURN DISTINCT b.name, b.className, b.returnType, b.parameters, b.body"
        )


# DB utility functions

def fetch_rows(conn, query: str, params: dict | None = None) -> list[tuple]:
    # Run a Cypher query and drain all rows into a list. Wraps KùzuDB's
    # has_next()/get_next() iteration pattern.
    result = conn.execute(query, params or {})
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


def row_to_method_info(row) -> MethodInfo:
    # Build MethodInfo from a Cypher row with columns:
    # (name, className, returnType, parameters, body).
    return MethodInfo(
        name=row[0], class_name=row[1],
        return_type=row[2] or "", parameters=row[3] or "",
        body=row[4] or "",
    )

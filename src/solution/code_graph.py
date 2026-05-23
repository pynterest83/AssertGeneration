import os
import re
import shutil
import logging
import weakref
from collections import defaultdict
from typing import Optional

import kuzu

from parser.multi_lang_parser import (
    MultiLanguageParser, FileExtractionResult,
    ExtractedClass, ExtractedMethod, ExtractedCall,
    ExtractedField, ExtractedHeritage,
)

logger = logging.getLogger(__name__)


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


def generate_id(label: str, qualname: str) -> str:
    return f"{label}:{qualname}"


def param_key(params: str) -> str:
    return re.sub(r'\s+', '', params).replace(',', '_') if params else ''


# CodeGraph class

class CodeGraph:
    # KùzuDB-backed code graph: tree-sitter parse → KùzuDB storage → Cypher search.

    def __init__(self, project_path: str, language: str = "java",
                 db_path: Optional[str] = None, force_reindex: bool = False):
        self.project_path = project_path
        self.language = language
        self.db_path = db_path or os.path.join(project_path, ".code_graph")

        if force_reindex:
            import glob
            for f in glob.glob(self.db_path + "*"):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

        import threading
        self._complete_marker = self.db_path + ".complete"
        already_built = os.path.isfile(self._complete_marker)
        self.db = kuzu.Database(self.db_path, read_only=already_built)
        self.thread_local = threading.local()
        self._all_conns = weakref.WeakSet()

        if not already_built:
            self.init_schema()
            if not self.is_indexed():
                self.build_graph()
            else:
                # DB already indexed from a previous run (marker file missing) — create it now
                with open(self._complete_marker, 'w'): pass

        # In-memory caches for fast lookup
        self._class_cache: dict[str, ClassInfo] = {}
        self._subclasses: dict[str, list[str]] = defaultdict(list)
        self.build_class_cache()

    @property
    def conn(self):
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = kuzu.Connection(self.db)
            self._all_conns.add(self.thread_local.conn)
        return self.thread_local.conn

    def close_all(self):
        for c in list(self._all_conns):
            try:
                c.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close_all()
        except Exception:
            pass

    # Schema

    def init_schema(self):
        for sql in SCHEMA_SQL:
            try:
                self.conn.execute(sql)
            except Exception as e:
                # Table might already exist
                if "already exists" not in str(e).lower():
                    logger.warning("Schema SQL failed: %s — %s", sql[:60], e)

    def is_indexed(self) -> bool:
        # Returns True if sentinel method node exists in DB.
        try:
            result = self.conn.execute(
                "MATCH (m:Method {name: '__indexed_complete__', className: '__sentinel__'}) "
                "RETURN count(m)"
            )
            row = result.get_next()
            return row[0] > 0
        except Exception:
            return False

    # Graph building

    def build_graph(self):
        from tqdm import tqdm

        parser = MultiLanguageParser()
        logger.info("Indexing %s (%s)...", self.project_path, self.language)

        extractions = parser.parse_project(
            self.project_path, self.language,
            on_progress=lambda cur, tot, fp: None,
        )

        # Collect all data first
        all_classes: list[ExtractedClass] = []
        all_methods: list[ExtractedMethod] = []
        all_calls: list[ExtractedCall] = []
        all_fields: list[ExtractedField] = []
        all_heritage: list[ExtractedHeritage] = []

        for ext in extractions:
            all_classes.extend(ext.classes)
            all_methods.extend(ext.methods)
            all_calls.extend(ext.calls)
            all_fields.extend(ext.fields)
            all_heritage.extend(ext.heritage)

        # Insert Class nodes
        for cls in tqdm(all_classes, desc="Classes", disable=not all_classes):
            cid = generate_id("Class", f"{cls.file_path}::{cls.name}")
            try:
                self.conn.execute(
                    "CREATE (c:Class {id: $id, name: $name, filePath: $fp, "
                    "startLine: $sl, endLine: $el})",
                    {"id": cid, "name": cls.name, "fp": cls.file_path,
                     "sl": cls.start_line, "el": cls.end_line}
                )
            except Exception as e:
                logger.debug("skip duplicate class: %s", e)

        # Insert Method nodes
        for m in tqdm(all_methods, desc="Methods", disable=not all_methods):
            mid = generate_id("Method", f"{m.class_name}:{m.name}:{param_key(m.parameters)}")
            try:
                self.conn.execute(
                    "CREATE (m:Method {id: $id, name: $name, className: $cn, "
                    "filePath: $fp, body: $body, returnType: $rt, parameters: $params, "
                    "startLine: $sl, endLine: $el})",
                    {"id": mid, "name": m.name, "cn": m.class_name,
                     "fp": m.file_path, "body": m.body, "rt": m.return_type,
                     "params": m.parameters, "sl": m.start_line, "el": m.end_line}
                )
            except Exception as e:
                logger.debug("skip duplicate method: %s", e)

        # Insert Field nodes
        for f in tqdm(all_fields, desc="Fields", disable=not all_fields):
            fid = generate_id("Field", f"{f.class_name}:{f.name}")
            try:
                self.conn.execute(
                    "CREATE (f:Field {id: $id, name: $name, fieldType: $ft, "
                    "className: $cn, modifier: $mod})",
                    {"id": fid, "name": f.name, "ft": f.field_type,
                     "cn": f.class_name, "mod": f.modifier}
                )
            except Exception as e:
                logger.debug("skip duplicate field: %s", e)

        # Insert HAS_METHOD edges
        for m in all_methods:
            if m.class_name:
                cid = generate_id("Class", f"{m.file_path}::{m.class_name}")
                mid = generate_id("Method", f"{m.class_name}:{m.name}:{param_key(m.parameters)}")
                try:
                    self.conn.execute(
                        "MATCH (c:Class {id: $cid}), (m:Method {id: $mid}) "
                        "CREATE (c)-[:HAS_METHOD]->(m)",
                        {"cid": cid, "mid": mid}
                    )
                except Exception as e:
                    logger.debug("skip HAS_METHOD %s->%s: %s", cid, mid, e)

        # Insert CALLS edges
        for call in all_calls:
            if not call.caller_method or not call.callee_name:
                continue
            try:
                cr = self.conn.execute(
                    "MATCH (m:Method {className: $cn, name: $mn}) "
                    "WHERE m.filePath = $fp RETURN m.id LIMIT 1",
                    {"cn": call.caller_class, "mn": call.caller_method, "fp": call.file_path}
                )
                crow = cr.get_next()
                if not crow:
                    continue
                caller_id = crow[0]
            except Exception as e:
                logger.debug("caller lookup failed: %s", e)
                continue
            try:
                callee_res = self.conn.execute(
                    "MATCH (m:Method {name: $name, className: $cn}) RETURN m.id LIMIT 1",
                    {"name": call.callee_name, "cn": call.caller_class}
                )
                crow2 = callee_res.get_next()
                if not crow2:
                    callee_res = self.conn.execute(
                        "MATCH (m:Method {name: $name}) "
                        "WHERE m.className <> '__sentinel__' RETURN m.id LIMIT 1",
                        {"name": call.callee_name}
                    )
                    crow2 = callee_res.get_next()
                if not crow2:
                    continue
                callee_id = crow2[0]
                exists = self.conn.execute(
                    "MATCH (a:Method {id: $aid})-[:CALLS]->(b:Method {id: $bid}) RETURN count(*)",
                    {"aid": caller_id, "bid": callee_id}
                )
                if exists.get_next()[0] == 0:
                    self.conn.execute(
                        "MATCH (a:Method {id: $aid}), (b:Method {id: $bid}) "
                        "CREATE (a)-[:CALLS]->(b)",
                        {"aid": caller_id, "bid": callee_id}
                    )
            except Exception as e:
                logger.debug("skip CALLS edge: %s", e)

        # Insert EXTENDS / IMPLEMENTS edges
        class_file_map: dict[str, str] = {cls.name: cls.file_path for cls in all_classes}

        for h in all_heritage:
            src_fp = class_file_map.get(h.class_name, "")
            if not src_fp:
                continue  # source class not in project — skip
            src_id = generate_id("Class", f"{src_fp}::{h.class_name}")

            if h.extends:
                tgt_fp = class_file_map.get(h.extends, "")
                if tgt_fp:
                    tgt_id = generate_id("Class", f"{tgt_fp}::{h.extends}")
                    try:
                        self.conn.execute(
                            "MATCH (a:Class {id: $aid}), (b:Class {id: $bid}) "
                            "CREATE (a)-[:EXTENDS]->(b)",
                            {"aid": src_id, "bid": tgt_id}
                        )
                    except Exception as e:
                        logger.debug("skip EXTENDS %s->%s: %s", src_id, tgt_id, e)
                # parent not in project (external) — no edge to create

            if h.implements:
                tgt_fp = class_file_map.get(h.implements, "")
                if tgt_fp:
                    tgt_id = generate_id("Class", f"{tgt_fp}::{h.implements}")
                    try:
                        self.conn.execute(
                            "MATCH (a:Class {id: $aid}), (b:Class {id: $bid}) "
                            "CREATE (a)-[:IMPLEMENTS]->(b)",
                            {"aid": src_id, "bid": tgt_id}
                        )
                    except Exception as e:
                        logger.debug("skip IMPLEMENTS %s->%s: %s", src_id, tgt_id, e)

        # Insert HAS_FIELD edges
        for f in all_fields:
            if f.class_name:
                cid = generate_id("Class", f"{f.file_path}::{f.class_name}")
                fid = generate_id("Field", f"{f.class_name}:{f.name}")
                try:
                    self.conn.execute(
                        "MATCH (c:Class {id: $cid}), (f:Field {id: $fid}) "
                        "CREATE (c)-[:HAS_FIELD]->(f)",
                        {"cid": cid, "fid": fid}
                    )
                except Exception as e:
                    logger.debug("skip HAS_FIELD %s->%s: %s", cid, fid, e)

        try:
            self.conn.execute(
                "CREATE (m:Method {id: 'sentinel:__indexed_complete__', "
                "name: '__indexed_complete__', className: '__sentinel__', "
                "filePath: '', body: '', returnType: '', parameters: '', "
                "startLine: 0, endLine: 0})"
            )
        except Exception as e:
            logger.debug("sentinel already exists: %s", e)

        # Count (exclude sentinel)
        try:
            res = self.conn.execute(
                "MATCH (m:Method) WHERE m.className <> '__sentinel__' RETURN count(m)"
            )
            method_count = res.get_next()[0]
            res = self.conn.execute("MATCH (c:Class) RETURN count(c)")
            class_count = res.get_next()[0]
            logger.info("Indexed %d classes, %d methods.", class_count, method_count)
        except Exception as e:
            logger.debug("count query failed: %s", e)

        # Write completion marker so parallel processes can open DB read-only
        with open(self._complete_marker, 'w'): pass

    # Class cache (for inheritance resolution)

    def build_class_cache(self):
        try:
            result = self.conn.execute("MATCH (c:Class) RETURN c.name")
            while result.has_next():
                row = result.get_next()
                name = row[0]
                ci = ClassInfo(name)
                self._class_cache[name] = ci
        except Exception:
            return

        # Populate extends/implements
        try:
            result = self.conn.execute(
                "MATCH (a:Class)-[:EXTENDS]->(b:Class) RETURN a.name, b.name"
            )
            while result.has_next():
                row = result.get_next()
                child, parent = row[0], row[1]
                if child in self._class_cache:
                    self._class_cache[child].extends.append(parent)
                self._subclasses[parent].append(child)
        except Exception:
            pass

        try:
            result = self.conn.execute(
                "MATCH (a:Class)-[:IMPLEMENTS]->(b:Class) RETURN a.name, b.name"
            )
            while result.has_next():
                row = result.get_next()
                child, iface = row[0], row[1]
                if child in self._class_cache:
                    self._class_cache[child].implements.append(iface)
        except Exception:
            pass

        # Populate fields
        try:
            result = self.conn.execute(
                "MATCH (c:Class)-[:HAS_FIELD]->(f:Field) "
                "RETURN c.name, f.name, f.fieldType, f.modifier"
            )
            while result.has_next():
                row = result.get_next()
                cname, fname, ftype, fmod = row[0], row[1], row[2], row[3]
                if cname in self._class_cache:
                    self._class_cache[cname].fields[fname] = ftype
                    self._class_cache[cname].field_modifiers[fname] = fmod
        except Exception:
            pass

    # Public search API

    def search(self, class_name: str = None, method_name: str = None,
               max_results: int = 10, standalone_only: bool = False) -> list[MethodInfo]:
        if class_name and method_name:
            return self.resolve_method(class_name, method_name, max_results)
        elif class_name:
            return self.search_by_class(class_name, max_results)
        elif method_name:
            return self.search_by_name(method_name, max_results, standalone_only=standalone_only)
        return []

    def search_by_class(self, class_name: str, limit: int) -> list[MethodInfo]:
        try:
            result = self.conn.execute(
                "MATCH (m:Method {className: $cn}) "
                "WHERE m.className <> '__sentinel__' "
                "RETURN m.name, m.className, m.returnType, m.parameters, m.body "
                "LIMIT $limit",
                {"cn": class_name, "limit": limit}
            )
            methods = []
            while result.has_next():
                row = result.get_next()
                methods.append(MethodInfo(
                    name=row[0], class_name=row[1], return_type=row[2] or "",
                    parameters=row[3] or "", body=row[4] or ""
                ))
            return methods
        except Exception as e:
            logger.warning("search_by_class failed: %s", e)
            return []

    def search_by_name(self, method_name: str, limit: int, standalone_only: bool = False) -> list[MethodInfo]:
        try:
            if standalone_only:
                result = self.conn.execute(
                    "MATCH (m:Method {name: $name}) "
                    "WHERE m.className = '' "
                    "RETURN m.name, m.className, m.returnType, m.parameters, m.body "
                    "LIMIT $limit",
                    {"name": method_name, "limit": limit}
                )
            else:
                result = self.conn.execute(
                    "MATCH (m:Method {name: $name}) "
                    "WHERE m.className <> '__sentinel__' "
                    "RETURN m.name, m.className, m.returnType, m.parameters, m.body "
                    "LIMIT $limit",
                    {"name": method_name, "limit": limit}
                )
            methods = []
            while result.has_next():
                row = result.get_next()
                methods.append(MethodInfo(
                    name=row[0], class_name=row[1], return_type=row[2] or "",
                    parameters=row[3] or "", body=row[4] or ""
                ))
            return methods
        except Exception as e:
            logger.warning("search_by_name failed: %s", e)
            return []

    def resolve_method(self, class_name: str, method_name: str,
                        limit: int) -> list[MethodInfo]:
        # Walk up parents then down subclasses if direct lookup fails.
        # Direct lookup
        results = self.search_class_method(class_name, method_name, limit)
        if results:
            return results

        # Walk up parent classes
        visited = {class_name}
        queue = list(self._class_cache.get(class_name, ClassInfo(class_name)).extends)

        while queue:
            parent = queue.pop(0)
            if parent in visited:
                continue
            visited.add(parent)

            results = self.search_class_method(parent, method_name, limit)
            if results:
                return results

            parent_info = self._class_cache.get(parent)
            if parent_info:
                queue.extend(parent_info.extends)

        # Walk down subclasses
        queue = list(self._subclasses.get(class_name, []))
        visited = {class_name}
        while queue:
            child = queue.pop(0)
            if child in visited:
                continue
            visited.add(child)

            results = self.search_class_method(child, method_name, limit)
            if results:
                return results

            queue.extend(self._subclasses.get(child, []))

        return []

    def search_class_method(self, class_name: str, method_name: str,
                             limit: int) -> list[MethodInfo]:
        try:
            result = self.conn.execute(
                "MATCH (m:Method {className: $cn, name: $name}) "
                "RETURN m.name, m.className, m.returnType, m.parameters, m.body "
                "LIMIT $limit",
                {"cn": class_name, "name": method_name, "limit": limit}
            )
            methods = []
            while result.has_next():
                row = result.get_next()
                methods.append(MethodInfo(
                    name=row[0], class_name=row[1], return_type=row[2] or "",
                    parameters=row[3] or "", body=row[4] or ""
                ))
            return methods
        except Exception:
            return []

    def search_with_callees(self, class_name: str, method_name: str,
                            depth: int = 1) -> list[MethodInfo]:
        # Single Cypher traversal returns target method + all callees up to N depth.
        # First get the target method (walks inheritance if needed)
        target = self.resolve_method(class_name, method_name, 1)
        if not target:
            return []

        results = list(target)
        actual = target[0]

        # Handles inherited methods whose actual class differs from requested class_name.
        try:
            id_res = self.conn.execute(
                "MATCH (m:Method {className: $cn, name: $mn}) RETURN m.id",
                {"cn": actual.class_name, "mn": actual.name}
            )
            target_ids = []
            while id_res.has_next():
                target_ids.append(id_res.get_next()[0])
        except Exception:
            return results

        if not target_ids:
            return results

        # Traverse callees for every matching overload
        for target_id in target_ids:
            try:
                result = self.conn.execute(
                    f"MATCH (a:Method {{id: $aid}})-[:CALLS*1..{depth}]->(b:Method) "
                    "WHERE b.className <> '__sentinel__' "
                    "RETURN DISTINCT b.name, b.className, b.returnType, b.parameters, b.body",
                    {"aid": target_id}
                )
                while result.has_next():
                    row = result.get_next()
                    mi = MethodInfo(
                        name=row[0], class_name=row[1], return_type=row[2] or "",
                        parameters=row[3] or "", body=row[4] or ""
                    )
                    if not any(r.name == mi.name and r.class_name == mi.class_name for r in results):
                        results.append(mi)
            except Exception as e:
                logger.debug("search_with_callees traversal: %s", e)

        return results

    # Class info API

    def get_class_info(self, class_name: str) -> list[ClassInfo]:
        # Returns list of 0 or 1 element (single dict lookup).
        ci = self._class_cache.get(class_name)
        return [ci] if ci else []


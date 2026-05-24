import os
import shutil
import logging
import weakref
from collections import defaultdict
from typing import Optional

import kuzu

from parser.multi_lang_parser import MultiLanguageParser
from helpers.db_utils import (
    MethodInfo, ClassInfo,
    SCHEMA_SQL, Queries,
    fetch_rows, row_to_method_info,
)
from helpers.build_utils import GraphBuilder

logger = logging.getLogger(__name__)


class CodeGraph:
    # KùzuDB-backed code graph: tree-sitter parse → KùzuDB storage → Cypher search.

    def __init__(self, project_path: str, language: str = "java",
                 db_path: Optional[str] = None, force_reindex: bool = False,
                 on_progress=None):
        self.project_path = project_path
        self.language = language
        self.db_path = db_path or os.path.join(project_path, ".code_graph")
        self._on_progress = on_progress

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
            result = self.conn.execute(Queries.CHECK_INDEXED_SENTINEL)
            row = result.get_next()
            return row[0] > 0
        except Exception:
            return False

    # Graph building

    def build_graph(self):
        parser = MultiLanguageParser()
        logger.info("Indexing %s (%s)...", self.project_path, self.language)
        extractions = parser.parse_project(
            self.project_path, self.language,
            on_progress=self._on_progress or (lambda cur, tot, fp: None),
        )
        GraphBuilder(self.conn).build(extractions)
        # Write completion marker so parallel processes can open DB read-only
        with open(self._complete_marker, 'w'): pass

    # Class cache (for inheritance resolution)

    def build_class_cache(self):
        try:
            for row in fetch_rows(self.conn, Queries.LOAD_ALL_CLASSES):
                name = row[0]
                self._class_cache[name] = ClassInfo(name)
        except Exception:
            return

        # Populate extends/implements
        try:
            for row in fetch_rows(self.conn, Queries.LOAD_EXTENDS_EDGES):
                child, parent = row[0], row[1]
                if child in self._class_cache:
                    self._class_cache[child].extends.append(parent)
                self._subclasses[parent].append(child)
        except Exception:
            pass

        try:
            for row in fetch_rows(self.conn, Queries.LOAD_IMPLEMENTS_EDGES):
                child, iface = row[0], row[1]
                if child in self._class_cache:
                    self._class_cache[child].implements.append(iface)
        except Exception:
            pass

        # Populate fields
        try:
            for row in fetch_rows(self.conn, Queries.LOAD_HAS_FIELD_EDGES):
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
            rows = fetch_rows(self.conn, Queries.SEARCH_BY_CLASS, {"cn": class_name, "limit": limit})
            return [row_to_method_info(r) for r in rows]
        except Exception as e:
            logger.warning("search_by_class failed: %s", e)
            return []

    def search_by_name(self, method_name: str, limit: int, standalone_only: bool = False) -> list[MethodInfo]:
        try:
            query = Queries.SEARCH_BY_NAME_STANDALONE if standalone_only else Queries.SEARCH_BY_NAME
            rows = fetch_rows(self.conn, query, {"name": method_name, "limit": limit})
            return [row_to_method_info(r) for r in rows]
        except Exception as e:
            logger.warning("search_by_name failed: %s", e)
            return []

    def resolve_method(self, class_name: str, method_name: str,
                        limit: int) -> list[MethodInfo]:
        # Walk up parents then down subclasses if direct lookup fails.
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
            rows = fetch_rows(self.conn, Queries.SEARCH_CLASS_METHOD, {
                "cn": class_name, "name": method_name, "limit": limit,
            })
            return [row_to_method_info(r) for r in rows]
        except Exception:
            return []

    def search_with_callees(self, class_name: str, method_name: str,
                            depth: int = 1) -> list[MethodInfo]:
        # Single Cypher traversal returns target method + all callees up to N depth.
        target = self.resolve_method(class_name, method_name, 1)
        if not target:
            return []

        results = list(target)
        actual = target[0]

        # Handles inherited methods whose actual class differs from requested class_name.
        try:
            id_rows = fetch_rows(self.conn, Queries.LOOKUP_METHOD_IDS_BY_CLASS_NAME, {
                "cn": actual.class_name, "mn": actual.name,
            })
            target_ids = [r[0] for r in id_rows]
        except Exception:
            return results

        if not target_ids:
            return results

        # Traverse callees for every matching overload
        for target_id in target_ids:
            try:
                rows = fetch_rows(self.conn, Queries.search_callees(depth), {"aid": target_id})
                for row in rows:
                    mi = row_to_method_info(row)
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

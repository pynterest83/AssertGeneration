import logging

from tqdm import tqdm

from .db_utils import Queries, generate_id, param_key

logger = logging.getLogger(__name__)


# ID helpers — only used during build

def class_id(file_path: str, name: str) -> str:
    return generate_id("Class", f"{file_path}::{name}")


def method_id(class_name: str, name: str, parameters: str) -> str:
    return generate_id("Method", f"{class_name}:{name}:{param_key(parameters)}")


def field_id(class_name: str, name: str) -> str:
    return generate_id("Field", f"{class_name}:{name}")


class GraphBuilder:
    # Inserts parsed extractions into KùzuDB. One instance per build_graph() call.

    def __init__(self, conn):
        self.conn = conn

    def build(self, extractions):
        classes, methods, calls, fields, heritage = self.aggregate(extractions)
        self.insert_class_nodes(classes)
        self.insert_method_nodes(methods)
        self.insert_field_nodes(fields)
        self.insert_has_method_edges(methods)
        self.insert_call_edges(calls)
        self.insert_heritage_edges(heritage, classes)
        self.insert_has_field_edges(fields)
        self.write_sentinel()
        self.log_counts()

    def aggregate(self, extractions):
        # Flatten per-file FileExtractionResult into combined lists.
        classes, methods, calls, fields, heritage = [], [], [], [], []
        for ext in extractions:
            classes.extend(ext.classes)
            methods.extend(ext.methods)
            calls.extend(ext.calls)
            fields.extend(ext.fields)
            heritage.extend(ext.heritage)
        return classes, methods, calls, fields, heritage

    def insert_class_nodes(self, classes):
        for cls in tqdm(classes, desc="Classes", disable=not classes):
            cid = class_id(cls.file_path, cls.name)
            try:
                self.conn.execute(Queries.INSERT_CLASS, {
                    "id": cid, "name": cls.name, "fp": cls.file_path,
                    "sl": cls.start_line, "el": cls.end_line,
                })
            except Exception as e:
                logger.debug("skip duplicate class: %s", e)

    def insert_method_nodes(self, methods):
        for m in tqdm(methods, desc="Methods", disable=not methods):
            mid = method_id(m.class_name, m.name, m.parameters)
            try:
                self.conn.execute(Queries.INSERT_METHOD, {
                    "id": mid, "name": m.name, "cn": m.class_name,
                    "fp": m.file_path, "body": m.body, "rt": m.return_type,
                    "params": m.parameters, "sl": m.start_line, "el": m.end_line,
                })
            except Exception as e:
                logger.debug("skip duplicate method: %s", e)

    def insert_field_nodes(self, fields):
        for f in tqdm(fields, desc="Fields", disable=not fields):
            fid = field_id(f.class_name, f.name)
            try:
                self.conn.execute(Queries.INSERT_FIELD, {
                    "id": fid, "name": f.name, "ft": f.field_type,
                    "cn": f.class_name, "mod": f.modifier,
                })
            except Exception as e:
                logger.debug("skip duplicate field: %s", e)

    def insert_has_method_edges(self, methods):
        for m in methods:
            if not m.class_name:
                continue
            cid = class_id(m.file_path, m.class_name)
            mid = method_id(m.class_name, m.name, m.parameters)
            try:
                self.conn.execute(Queries.INSERT_HAS_METHOD_EDGE, {"cid": cid, "mid": mid})
            except Exception as e:
                logger.debug("skip HAS_METHOD %s->%s: %s", cid, mid, e)

    def insert_call_edges(self, calls):
        for call in calls:
            # Đầu tiên bỏ qua call thiếu caller hoặc callee.
            if not call.caller_method or not call.callee_name:
                continue
            try:
                # find caller_id
                cr = self.conn.execute(Queries.LOOKUP_CALLER_BY_FILEPATH, {
                    "cn": call.caller_class, "mn": call.caller_method, "fp": call.file_path,
                })
                if not cr.has_next():
                    continue
                caller_id = cr.get_next()[0]
            except Exception as e:
                logger.debug("caller lookup failed: %s", e)
                continue
            try:
                #  Sau khi có caller, builder tìm callee, ưu tiên tìm method cùng class trước:
                callee_res = self.conn.execute(Queries.LOOKUP_CALLEE_IN_SAME_CLASS, {
                    "name": call.callee_name, "cn": call.caller_class,
                })
                if callee_res.has_next():
                    callee_id = callee_res.get_next()[0]
                else:
                    # Nếu không có, nó tìm bất kỳ class nào có method tên đó
                    callee_res = self.conn.execute(Queries.LOOKUP_CALLEE_ANY_CLASS, {"name": call.callee_name})
                    if not callee_res.has_next():
                        continue
                    callee_id = callee_res.get_next()[0]
                # Nếu tìm được callee, nó kiểm tra edge đã tồn tại chưa
                exists = self.conn.execute(Queries.CHECK_CALLS_EDGE_EXISTS, {"aid": caller_id, "bid": callee_id})
                if exists.get_next()[0] == 0:
                    self.conn.execute(Queries.INSERT_CALLS_EDGE, {"aid": caller_id, "bid": callee_id})
            except Exception as e:
                logger.debug("skip CALLS edge: %s", e)

    def insert_heritage_edges(self, heritage, classes):
        # Parser có thể tạo: ExtractedHeritage(class_name="Dog", extends="Animal")
        # add path to class to create ID
        class_file_map: dict[str, str] = {cls.name: cls.file_path for cls in classes}

        for h in heritage:
            src_fp = class_file_map.get(h.class_name, "")
            if not src_fp:
                continue  # source class not in project — skip
            src_id = class_id(src_fp, h.class_name)
            # add edges
            if h.extends:
                tgt_fp = class_file_map.get(h.extends, "")
                if tgt_fp:
                    tgt_id = class_id(tgt_fp, h.extends)
                    try:
                        self.conn.execute(Queries.INSERT_EXTENDS_EDGE, {"aid": src_id, "bid": tgt_id})
                    except Exception as e:
                        logger.debug("skip EXTENDS %s->%s: %s", src_id, tgt_id, e)
            # add implements
            if h.implements:
                tgt_fp = class_file_map.get(h.implements, "")
                if tgt_fp:
                    tgt_id = class_id(tgt_fp, h.implements)
                    try:
                        self.conn.execute(Queries.INSERT_IMPLEMENTS_EDGE, {"aid": src_id, "bid": tgt_id})
                    except Exception as e:
                        logger.debug("skip IMPLEMENTS %s->%s: %s", src_id, tgt_id, e)

    def insert_has_field_edges(self, fields):
        # add field to class
        for f in fields:
            if not f.class_name:
                continue
            cid = class_id(f.file_path, f.class_name)
            fid = field_id(f.class_name, f.name)
            try:
                self.conn.execute(Queries.INSERT_HAS_FIELD_EDGE, {"cid": cid, "fid": fid})
            except Exception as e:
                logger.debug("skip HAS_FIELD %s->%s: %s", cid, fid, e)
    # method to check if build done
    def write_sentinel(self):
        try:
            self.conn.execute(Queries.INSERT_SENTINEL_METHOD)
        except Exception as e:
            logger.debug("sentinel already exists: %s", e)

    def log_counts(self):
        try:
            res = self.conn.execute(Queries.COUNT_METHODS_EXCLUDING_SENTINEL)
            method_count = res.get_next()[0]
            res = self.conn.execute(Queries.COUNT_CLASSES)
            class_count = res.get_next()[0]
            logger.info("Indexed %d classes, %d methods.", class_count, method_count)
        except Exception as e:
            logger.debug("count query failed: %s", e)

"""Export KùzuDB CodeGraph into JSON suitable for vis-network frontend.

Frontend (graphPanelProvider.ts) expects:
  GraphNode: { id, label, type: 'Class'|'Method'|'Field', filePath?, body? }
  GraphEdge: { from, to, type: 'CALLS'|'EXTENDS'|'IMPLEMENTS'|'HAS_METHOD'|'HAS_FIELD' }

Limits to NODE_LIMIT total nodes to keep the UI responsive.
"""
import logging

log = logging.getLogger("assertgen")

NODE_LIMIT = 1000


def _iter_rows(result):
    while result.has_next():
        yield result.get_next()


def export_graph_json(code_graph) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []
    conn = code_graph.conn

    # ── Classes ────────────────────────────────────────────────────────────
    try:
        res = conn.execute(
            "MATCH (c:Class) RETURN c.id, c.name, c.filePath "
            f"LIMIT {NODE_LIMIT}"
        )
        for row in _iter_rows(res):
            nodes.append({
                "id": row[0],
                "label": row[1],
                "type": "Class",
                "filePath": row[2] or "",
            })
    except Exception as e:
        log.warning("export_graph_json: Class query failed: %s", e)

    remaining = NODE_LIMIT - len(nodes)

    # ── Methods (exclude sentinel) ─────────────────────────────────────────
    if remaining > 0:
        try:
            res = conn.execute(
                "MATCH (m:Method) WHERE m.className <> '__sentinel__' "
                "RETURN m.id, m.name, m.filePath, m.body "
                f"LIMIT {remaining}"
            )
            for row in _iter_rows(res):
                nodes.append({
                    "id": row[0],
                    "label": row[1],
                    "type": "Method",
                    "filePath": row[2] or "",
                    "body": row[3] or "",
                })
        except Exception as e:
            log.warning("export_graph_json: Method query failed: %s", e)

    remaining = NODE_LIMIT - len(nodes)

    # ── Fields ─────────────────────────────────────────────────────────────
    if remaining > 0:
        try:
            res = conn.execute(
                "MATCH (f:Field) RETURN f.id, f.name "
                f"LIMIT {remaining}"
            )
            for row in _iter_rows(res):
                nodes.append({
                    "id": row[0],
                    "label": row[1],
                    "type": "Field",
                    "filePath": "",
                })
        except Exception as e:
            log.warning("export_graph_json: Field query failed: %s", e)

    node_ids = {n["id"] for n in nodes}

    edge_queries = [
        ("HAS_METHOD", "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) RETURN c.id, m.id"),
        ("CALLS",      "MATCH (a:Method)-[:CALLS]->(b:Method) RETURN a.id, b.id"),
        ("EXTENDS",    "MATCH (a:Class)-[:EXTENDS]->(b:Class) RETURN a.id, b.id"),
        ("IMPLEMENTS", "MATCH (a:Class)-[:IMPLEMENTS]->(b:Class) RETURN a.id, b.id"),
        ("HAS_FIELD",  "MATCH (c:Class)-[:HAS_FIELD]->(f:Field) RETURN c.id, f.id"),
    ]
    for edge_type, cypher in edge_queries:
        try:
            res = conn.execute(cypher)
            for row in _iter_rows(res):
                if row[0] in node_ids and row[1] in node_ids:
                    edges.append({"from": row[0], "to": row[1], "type": edge_type})
        except Exception as e:
            log.warning("export_graph_json: %s query failed: %s", edge_type, e)

    return {"nodes": nodes, "edges": edges}

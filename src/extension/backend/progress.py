"""Thread-safe progress event queue + helper used by both server.py and pipeline_runner.py.

Events pushed via push_progress() are:
  - emitted to the SSE /progress stream (one consumer at a time)
  - logged to the assertgen terminal logger with a short summary

The queue is shared module-level; only one pipeline runs at a time per process.
"""
import logging
import queue
from pathlib import Path

log = logging.getLogger("assertgen")

_progress_queue: "queue.Queue[dict]" = queue.Queue()


def get_queue() -> "queue.Queue[dict]":
    return _progress_queue


def push_progress(event: dict) -> None:
    """Enqueue an SSE event and log a short human-readable summary."""
    _progress_queue.put(event)

    t = event.get("type", "")
    if t == "stage":
        log.info("── STAGE: %s", event.get("message", event.get("stage")))
    elif t == "extraction":
        cur, tot = event.get("current", 0), event.get("total", 0)
        if tot > 0 and (cur == 1 or cur % 10 == 0 or cur == tot):
            log.info("  Extract  [%d/%d] %s", cur, tot, event.get("file", ""))
    elif t == "extraction_complete":
        log.info("  Extracted %d test cases", event.get("test_count", 0))
    elif t == "graph_building":
        cur, tot = event.get("current", 0), event.get("total", 0)
        if tot > 0 and (cur == 1 or cur % 10 == 0 or cur == tot):
            log.info("  Graph    [%d/%d] %s", cur, tot, Path(event.get("file", "")).name)
    elif t == "graph_building_complete":
        log.info("  Graph build complete")
    elif t == "inference_progress":
        cur, tot = event.get("current", 0), event.get("total", 0)
        log.info("  Infer    [%d/%d] %s", cur, tot, event.get("test_name", ""))
    elif t == "inference":
        agent = event.get("agent", "")
        test = event.get("test_name", "")
        log.info("    %-25s → %s", agent, test[:60])
    elif t == "inference_complete":
        log.info("  Inference complete: %d results", event.get("result_count", 0))
    elif t == "injection":
        log.info("  Injected → %s", event.get("file", ""))
    elif t == "injection_warning":
        log.warning("  Injection warning: %s", event.get("message", ""))
    elif t == "pipeline_complete":
        log.info("── DONE: %d tests, files: %s",
                 event.get("total_tests", 0), event.get("injected_files", []))
    elif t == "pipeline_error":
        log.error("── PIPELINE ERROR: %s", event.get("message", ""))
    elif t == "error":
        log.error("  Error: %s", event.get("message", ""))

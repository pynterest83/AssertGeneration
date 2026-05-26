import logging
import threading

from langchain_core.tools import tool, ToolException

from helpers.db_utils import format_class_header
from helpers.parsing_utils import normalize_args_from_kwargs

# Optional progress hook — present only when running under the VS Code extension.
try:
    from progress import push_progress as _push_progress  # type: ignore
except ImportError:  # standalone CLI run — no SSE consumer
    def _push_progress(_event: dict) -> None:
        pass


def is_quota_error(exc) -> bool:
    """True for HTTP 429 / quota-exhausted / auth errors from the LLM API."""
    try:
        import openai
        if isinstance(exc, (openai.RateLimitError, openai.AuthenticationError,
                            openai.PermissionDeniedError)):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return any(k in msg for k in ('quota', 'throttling', 'insufficient_quota',
                                   'balance', 'billing', 'rate limit', 'rate_limit',
                                   'credits', 'afford'))

logger = logging.getLogger(__name__)


def create_tools(code_graph, language: str = "java"):
    """Create tools backed by CodeGraph.

    Returns a list containing search_relevant_code. The tool exposes two helpers:
      tool.reset_counter(known_external=None)  — reset per-agent state
      tool.get_external_cache()                — frozenset of confirmed-external class names
    """
    # Closure dict + lock instead of threading.local(): LangGraph's ToolNode executes
    # parallel tool calls in separate ThreadPoolExecutor threads, each with its own
    # thread-local storage → count would reset to 0 per thread. Shared dict fixes that.
    state = {"query_counts": {}, "external_cache": set()}
    lock = threading.Lock()

    def reset_counter(known_external: set | None = None):
        """Reset per-agent tool state. Call once before each agent.invoke()."""
        with lock:
            state["query_counts"] = {}
            state["external_cache"] = set(known_external) if known_external else set()

    def get_external_cache() -> frozenset:
        """Return confirmed-external class names accumulated so far."""
        with lock:
            return frozenset(state["external_cache"])

    @tool
    def search_relevant_code(class_name: str = '', method_name: str = '',
                             include_callees: bool = False, **kwargs) -> str:
        """Look up project-internal source code from the code graph.

        Args:
          class_name:
            Class to search. Required for Java. Required for JavaScript method lookup.
            Omit only for Python standalone function lookup.
          method_name:
            Method/function name. If omitted, returns class header, fields, and
            method signatures (no bodies). Provide to get the full method body.
          include_callees:
            Expand one method with the full bodies of project-internal methods it calls.
            Use ONLY when: the method body contains calls to other classes in this project
            AND you need to understand their implementation to reason about return values or state.
            Do NOT use for: utility/logging methods, methods that only call external libraries,
            or when you already have callee info from a previous query.
            Only effective when both class_name and method_name are given.
            Use at most once per (class, method) pair.

        Language constraints:
          - Java: class_name is mandatory.
          - JavaScript: class_name is mandatory for method lookup.
          - Python: standalone function lookup may omit class_name.
        """
        class_name, method_name, include_callees, recovered, parse_failed = normalize_args_from_kwargs(
            class_name, method_name, include_callees, kwargs
        )
        if recovered:
            logger.debug(
                "Recovered malformed tool args -> class_name=%r method_name=%r include_callees=%r",
                class_name, method_name, include_callees
            )

        if not class_name and not method_name:
            if parse_failed:
                return (
                    "Error: malformed tool arguments. Use object keys exactly: "
                    "{'class_name': 'Foo', 'method_name': 'bar', 'include_callees': false}."
                )
            return "Error: either class_name or method_name is required."
        if not class_name:
            if language == "java":
                return ("Error: class_name is required for Java. "
                        "Provide the class that contains the method.")
            if method_name and language == "javascript":
                raise ToolException(
                    "Error: class_name is required for JavaScript method lookup. "
                    "Provide the class or module name. For top-level functions, "
                    "use the file name stem as class_name."
                )

        # Guards (order: ext_cache → repeat — cheapest first)
        check_key = class_name if class_name else method_name
        with lock:
            ext_cache = state["external_cache"]
            q_counts = state["query_counts"]

            if check_key in ext_cache:
                return (
                    f"Already confirmed external/not-in-project for '{check_key}'. "
                    "STOP ALL TOOL CALLS. You already have everything you need. "
                    "Write your final answer NOW — do not call any more tools."
                )

            qk = (class_name.strip(), method_name.strip())
            count = q_counts.get(qk, 0) + 1
            q_counts[qk] = count
            if count >= 2:
                return (
                    f"Already have result for '{class_name or method_name}' (queried {count} times). "
                    "Do NOT query this again. Continue with other queries or call finish()."
                )

        # Code-graph query (outside lock — can be slow)
        _push_progress({
            "type": "tool_call",
            "tool": "search_relevant_code",
            "class_name": class_name,
            "method_name": method_name,
            "include_callees": bool(include_callees),
        })
        if method_name and include_callees and class_name:
            results = code_graph.search_with_callees(class_name, method_name, depth=1)
        else:
            standalone_only = (not class_name and method_name and language == "python")
            results = code_graph.search(
                class_name=class_name or None,
                method_name=method_name or None,
                max_results=10,
                standalone_only=standalone_only,
            )

        parts = []
        # add classinfo
        if class_name:
            ci_list = code_graph.get_class_info(class_name)
            if ci_list:
                parts.append(format_class_header(ci_list[0]))

        if not results:
            # return class info if possible
            if parts:
                out = '\n'.join(parts)
                _push_progress({"type": "tool_result",
                                "tool": "search_relevant_code",
                                "result_count": 0,
                                "preview": out[:600]})
                return out
            query = (f"{class_name}.{method_name}" if class_name and method_name
                     else (class_name or method_name))
            msg = (f"No results found for '{query}'. This class/function is likely "
                   "from a standard or external library — its source is not in this "
                   "project. Do NOT query this class or any of its methods again.")
            cache_key = class_name if class_name else method_name
            with lock:
                state["external_cache"].add(cache_key)
            _push_progress({"type": "tool_result",
                            "tool": "search_relevant_code",
                            "result_count": 0,
                            "external": True,
                            "preview": f"(no results — '{cache_key}' marked external)"})
            return msg

        for mi in results:
            if method_name:
                parts.append(mi.format())
            else:
                parts.append(f"// {mi.signature()}")

        out = '\n---\n'.join(parts)
        _push_progress({"type": "tool_result",
                        "tool": "search_relevant_code",
                        "result_count": len(results),
                        "preview": out[:600]})
        return out

    # Expose helpers on the tool object. StructuredTool is a Pydantic v2 model that
    # blocks unknown fields via __setattr__; object.__setattr__ bypasses that check
    # and writes directly to __dict__
    object.__setattr__(search_relevant_code, 'reset_counter', reset_counter)
    object.__setattr__(search_relevant_code, 'get_external_cache', get_external_cache)

    return [search_relevant_code, finish_tool]


@tool("finish")
def finish_tool() -> str:
    """Signal that you are done searching. Call this only when ready to write your final answer. After calling this, output your answer as plain text — no more tool calls."""
    return "Finalized. Now write your complete answer as plain text. Do NOT call any more tools."


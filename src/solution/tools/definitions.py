import logging
import re
import threading

from langchain_core.tools import tool, ToolException


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

MALFORMED_ARG_RE = re.compile(
    r"""^\s*class_name\s*=\s*['"]([^'"]*)['"]\s*,\s*"""
    r"""method_name\s*=\s*['"]([^'"]*)['"]"""
    r"""(?:\s*,\s*include_callees\s*=\s*(True|False|true|false))?\s*$"""
)


def normalize_args_from_kwargs(class_name: str, method_name: str, include_callees: bool,
                                extra_kwargs: dict) -> tuple[str, str, bool, bool, bool]:
    combined_candidate = (class_name or method_name or "").strip()
    if combined_candidate:
        m = MALFORMED_ARG_RE.match(combined_candidate)
        if m:
            norm_include = include_callees
            if m.group(3) is not None:
                norm_include = m.group(3).lower() == "true"
            return m.group(1).strip(), m.group(2).strip(), norm_include, True, False
        if "class_name" in combined_candidate and "method_name" in combined_candidate:
            return "", "", include_callees, False, True

    if class_name or method_name or not extra_kwargs:
        return class_name, method_name, include_callees, False, False

    if len(extra_kwargs) != 1:
        return class_name, method_name, include_callees, False, False

    malformed_key, malformed_val = next(iter(extra_kwargs.items()))
    if malformed_val not in ("", None):
        return class_name, method_name, include_callees, False, False

    m = MALFORMED_ARG_RE.match(str(malformed_key))
    if not m:
        return class_name, method_name, include_callees, False, True

    norm_class = m.group(1).strip()
    norm_method = m.group(2).strip()
    norm_include = include_callees
    if m.group(3) is not None:
        norm_include = m.group(3).lower() == "true"
    return norm_class, norm_method, norm_include, True, False


def query_key(class_name: str, method_name: str) -> tuple[str, str]:
    return (class_name.strip(), method_name.strip())


def create_tools(code_graph, language: str = "java"):
    """Create tools backed by CodeGraph.

    Returns a list containing search_relevant_code. The tool exposes two helpers:
      tool.reset_counter(known_external=None)  — reset per-agent state
      tool.get_external_cache()                — frozenset of confirmed-external class names
    """
    # TH-01: closure dict + lock instead of threading.local().
    # LangGraph's ToolNode executes parallel tool calls in separate ThreadPoolExecutor
    # threads, each with its own thread-local storage → count resets to 0 per thread.
    # A closure dict shared across all threads (with lock) fixes this correctly.
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

            qk = query_key(class_name, method_name)
            count = q_counts.get(qk, 0) + 1
            q_counts[qk] = count
            if count >= 2:
                return (
                    f"Already have result for '{class_name or method_name}' (queried {count} times). "
                    "Do NOT query this again. Continue with other queries or call finish()."
                )

        # Code-graph query (outside lock — can be slow)
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
        if class_name:
            ci_list = code_graph.get_class_info(class_name)
            ci = ci_list[0] if ci_list else None
            if ci:
                header_parts = [f"class {ci.name}"]
                if ci.extends:
                    header_parts.append(f"extends {', '.join(ci.extends)}")
                if ci.implements:
                    header_parts.append(f"implements {', '.join(ci.implements)}")
                parts.append(' '.join(header_parts))
                if ci.fields:
                    fields_str = '\n'.join(
                        f"  {ci.field_modifiers.get(fname, 'package')} {ftype} {fname};"
                        for fname, ftype in ci.fields.items()
                    )
                    parts.append(f"Fields:\n{fields_str}")

        if not results:
            if parts:
                return '\n'.join(parts)
            query = (f"{class_name}.{method_name}" if class_name and method_name
                     else (class_name or method_name))
            msg = (f"No results found for '{query}'. This class/function is likely "
                   "from a standard or external library — its source is not in this "
                   "project. Do NOT query this class or any of its methods again.")
            cache_key = class_name if class_name else method_name
            with lock:
                state["external_cache"].add(cache_key)
            return msg

        for mi in results:
            if method_name:
                parts.append(mi.format())
            else:
                parts.append(f"// {mi.signature()}")

        return '\n---\n'.join(parts)

    # Expose helpers on the tool object. StructuredTool is a Pydantic v2 model that
    # blocks unknown fields via __setattr__; object.__setattr__ bypasses that check
    # and writes directly to __dict__ (Pydantic models don't use __slots__).
    object.__setattr__(search_relevant_code, 'reset_counter', reset_counter)
    object.__setattr__(search_relevant_code, 'get_external_cache', get_external_cache)

    return [search_relevant_code, finish_tool]


@tool("finish")
def finish_tool() -> str:
    """Signal that you are done searching. Call this only when ready to write your final answer. After calling this, output your answer as plain text — no more tool calls."""
    return "Finalized. Now write your complete answer as plain text. Do NOT call any more tools."


import logging
from typing import Any

from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, SystemMessage
from prompts.code_analyzer import CODE_ANALYZER_SYSTEM, CODE_ANALYZER_HUMAN
from schemas import CodeAnalysis
from state import MAX_AGENT_STEPS
from lang_config import LANG_CONFIGS
from tools.definitions import is_quota_error

STRUCTURED_RESPONSE_PROMPT = (
    "Based on your analysis above, produce a structured CodeAnalysis as JSON. "
    "Fill in every field based on what you discovered from the code and tool calls. "
    "Be specific and concrete."
)

logger = logging.getLogger(__name__)


def format_analysis(analysis: CodeAnalysis) -> str:
    parts = [f"Signature: {analysis.signature}"]
    if analysis.fields_summary:
        parts.append(f"Fields: {analysis.fields_summary}")
    if analysis.branches:
        parts.append("Branches:\n" + "\n".join(f"  - {b}" for b in analysis.branches))
    if analysis.return_conditions:
        parts.append("Return conditions:\n" + "\n".join(f"  - {r}" for r in analysis.return_conditions))
    if analysis.dependencies:
        parts.append("Dependencies:\n" + "\n".join(f"  - {d}" for d in analysis.dependencies))
    return "\n".join(parts)


def make_analyzer_node(llm, tools):
    # Cache is safe: make_analyzer_node is called once per build_graph() call, which is
    # fresh per sample. Each call creates a new closure with a new empty cache.
    agent_cache: dict[str, Any] = {}
    # streaming=False required — streaming=True hangs on structured output with SSE endpoints.
    extraction_llm = llm.model_copy(update={'streaming': False}).with_structured_output(CodeAnalysis)

    def node(state):
        tools[0].reset_counter()

        lang = state.get('language', 'java')
        lang_cfg = LANG_CONFIGS.get(lang, LANG_CONFIGS['java'])
        system_prompt = CODE_ANALYZER_SYSTEM.format(**lang_cfg)

        if lang not in agent_cache:
            agent_cache[lang] = create_react_agent(
                model=llm,
                tools=tools,
                prompt=system_prompt,
            )
        agent = agent_cache[lang]

        docstring_section = f"Docstring: {state['docstring']}" if state.get('docstring') else ""
        human_msg = CODE_ANALYZER_HUMAN.format(
            language=lang_cfg['language'],
            focal_class=state.get('focal_class', ''),
            focal_method=state['focal_method'],
            return_type=state.get('return_type', ''),
            docstring_section=docstring_section,
        )
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": human_msg}]},
                config={"recursion_limit": MAX_AGENT_STEPS},
            )
            known_external = list(tools[0].get_external_cache())
            messages = result["messages"]
            last_msg = messages[-1]
            last_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            try:
                # Send only the last AI message (not full history) to avoid 504 on large context
                structured = extraction_llm.invoke(
                    last_content + "\n\n" + STRUCTURED_RESPONSE_PROMPT
                )
                if isinstance(structured, CodeAnalysis):
                    return {'analysis': format_analysis(structured), 'known_external': known_external}
            except Exception as e2:
                logger.debug("CodeAnalyzer structured extraction failed: %s", e2)
            logger.debug("CodeAnalyzer fallback to raw content")
            return {'analysis': last_content or '', 'known_external': known_external}
        except GraphRecursionError as e:
            logger.warning("CodeAnalyzer hit recursion limit for %s", state.get('focal_class', '?'))
            known_external = list(tools[0].get_external_cache())
            try:
                plain_llm = llm.model_copy(update={'streaming': False})
                result = plain_llm.invoke([
                    SystemMessage(content=system_prompt + "\n\nAnalyze the provided code concisely. Do NOT call any tools."),
                    HumanMessage(content=human_msg),
                ])
                raw = result.content if hasattr(result, 'content') else str(result)
                if raw:
                    return {'analysis': raw, 'known_external': known_external}
            except Exception as e2:
                if is_quota_error(e2):
                    raise
                logger.warning("CodeAnalyzer fallback LLM call failed: %s", e2)
            return {'analysis': '', 'known_external': known_external}
        except Exception as e:
            if is_quota_error(e):
                raise
            logger.warning("CodeAnalyzer stopped early for %s: %s", state.get('focal_class', '?'), e)
            return {'analysis': '', 'known_external': list(tools[0].get_external_cache())}

    return node

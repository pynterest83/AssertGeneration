import logging

from langgraph.prebuilt import create_react_agent
from prompts.code_analyzer import CODE_ANALYZER_SYSTEM, CODE_ANALYZER_HUMAN
from schemas import CodeAnalysis
from state import MAX_AGENT_STEPS

_STRUCTURED_RESPONSE_PROMPT = (
    "Based on your analysis above, produce a structured CodeAnalysis. "
    "Fill in every field based on what you discovered from the code and tool calls. "
    "Be specific and concrete."
)

logger = logging.getLogger(__name__)


def _format_analysis(analysis: CodeAnalysis) -> str:
    """Format structured CodeAnalysis into a readable string for downstream agents."""
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
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=CODE_ANALYZER_SYSTEM,
        # Tuple format: (system_prompt_for_structured_step, schema)
        # This gives the generate_structured_response step its own system prompt
        response_format=(_STRUCTURED_RESPONSE_PROMPT, CodeAnalysis),
    )

    def node(state):
        docstring_section = f"Docstring: {state['docstring']}" if state.get('docstring') else ""
        human_msg = CODE_ANALYZER_HUMAN.format(
            focal_class=state.get('focal_class', ''),
            focal_method=state['focal_method'],
            return_type=state.get('return_type', ''),
            docstring_section=docstring_section,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": human_msg}]},
            config={"recursion_limit": MAX_AGENT_STEPS},
        )
        # Extract structured response
        try:
            structured = result.get("structured_response")
            if isinstance(structured, CodeAnalysis):
                return {'analysis': _format_analysis(structured)}
        except Exception:
            pass
        # Fallback: use raw content
        last_msg = result["messages"][-1]
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        logger.debug("CodeAnalyzer fallback to raw content")
        return {'analysis': content or ''}

    return node

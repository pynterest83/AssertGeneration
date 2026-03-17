import logging

from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, SystemMessage
from prompts.state_predictor import STATE_PREDICTOR_SYSTEM, STATE_PREDICTOR_HUMAN
from schemas import StatePrediction
from state import MAX_AGENT_STEPS
from tools.definitions import reset_tool_call_counter

_STRUCTURED_RESPONSE_PROMPT = (
    "Based on your execution trace above, produce a structured StatePrediction. "
    "List concrete variable values, clearly identify what is observable vs inaccessible (private with no getter), "
    "and recommend the best assertion target."
)

logger = logging.getLogger(__name__)


def _format_prediction(pred: StatePrediction) -> str:
    """Format structured StatePrediction into a readable string for the assertion generator."""
    parts = []
    if pred.variable_states:
        parts.append("Variable states:\n" + "\n".join(f"  - {v}" for v in pred.variable_states))
    parts.append(f"Observable state: {pred.observable_state}")
    if pred.inaccessible:
        parts.append("Inaccessible (private, no getter):\n" + "\n".join(f"  - {f}" for f in pred.inaccessible))
    parts.append(f"Assertion target: {pred.assertion_target}")
    return "\n".join(parts)


def make_predictor_node(llm, tools):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=STATE_PREDICTOR_SYSTEM,
        response_format=(_STRUCTURED_RESPONSE_PROMPT, StatePrediction),
    )

    def node(state):
        reset_tool_call_counter()  # Fresh budget for this agent
        human_msg = STATE_PREDICTOR_HUMAN.format(
            focal_class=state.get('focal_class', ''),
            analysis=state.get('analysis', ''),
            focal_method=state['focal_method'],
            test_prefix=state['test_prefix'],
        )
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": human_msg}]},
                config={"recursion_limit": MAX_AGENT_STEPS},
            )
            # Extract structured response
            try:
                structured = result.get("structured_response")
                if isinstance(structured, StatePrediction):
                    return {'prediction': _format_prediction(structured)}
            except Exception:
                pass
            # Fallback: use raw content
            last_msg = result["messages"][-1]
            content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            logger.debug("StatePredictor fallback to raw content")
            return {'prediction': content or ''}
        except GraphRecursionError:
            logger.warning("StatePredictor hit recursion limit for %s, using direct LLM fallback",
                           state.get('focal_class', '?'))
            try:
                fallback_llm = llm.with_structured_output(StatePrediction)
                fallback_result = fallback_llm.invoke([
                    SystemMessage(content=STATE_PREDICTOR_SYSTEM + "\n\nProduce your prediction based ONLY on the code and analysis provided. Do NOT request any tool calls."),
                    HumanMessage(content=human_msg),
                ])
                if isinstance(fallback_result, StatePrediction):
                    return {'prediction': _format_prediction(fallback_result)}
            except Exception as e2:
                logger.warning("StatePredictor structured fallback also failed: %s", e2)
            return {'prediction': ''}

    return node

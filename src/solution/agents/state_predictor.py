import logging

from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, SystemMessage
from prompts.state_predictor import STATE_PREDICTOR_SYSTEM, STATE_PREDICTOR_HUMAN
from schemas import StatePrediction
from state import MAX_AGENT_STEPS
from lang_config import LANG_CONFIGS
from tools.definitions import is_quota_error

STRUCTURED_RESPONSE_PROMPT = (
    "Based on your execution trace above, produce a structured StatePrediction as JSON. "
    "List concrete variable values, clearly identify what is observable vs inaccessible (private with no getter), "
    "and recommend the best assertion target."
)

logger = logging.getLogger(__name__)


def format_prediction(pred: StatePrediction) -> str:
    parts = []
    if pred.variable_states:
        parts.append("Variable states:\n" + "\n".join(f"  - {v}" for v in pred.variable_states))
    parts.append(f"Observable state: {pred.observable_state}")
    if pred.inaccessible:
        parts.append("Inaccessible (private, no getter):\n" + "\n".join(f"  - {f}" for f in pred.inaccessible))
    parts.append(f"Assertion target: {pred.assertion_target}")
    return "\n".join(parts)


def make_predictor_node(llm, tools):
    # streaming=False required — streaming=True hangs on structured output with SSE endpoints.
    extraction_llm = llm.model_copy(update={'streaming': False}).with_structured_output(StatePrediction)

    def node(state):
        lang = state.get('language', 'java')
        focal_class = state.get('focal_class', '')

        known_ext = set(state.get('known_external') or []) | ({focal_class} if focal_class else set())
        tools[0].reset_counter(known_external=known_ext)

        lang_cfg = LANG_CONFIGS.get(lang, LANG_CONFIGS['java'])
        system_prompt = STATE_PREDICTOR_SYSTEM.format(**lang_cfg)

        agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

        known_ext_display = sorted(known_ext - {focal_class})
        known_external_line = (
            ', '.join(known_ext_display) if known_ext_display else '(none identified so far)'
        )
        human_msg = STATE_PREDICTOR_HUMAN.format(
            focal_class=focal_class,
            analysis=state.get('analysis', ''),
            known_external_line=known_external_line,
            test_prefix=state['test_prefix'],
        )
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": human_msg}]},
                config={"recursion_limit": MAX_AGENT_STEPS},
            )
            messages = result["messages"]
            last_msg = messages[-1]
            last_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            try:
                # Send only the last AI message (not full history) to avoid 504 on large context
                structured = extraction_llm.invoke(
                    last_content + "\n\n" + STRUCTURED_RESPONSE_PROMPT
                )
                if isinstance(structured, StatePrediction):
                    return {'prediction': format_prediction(structured)}
            except Exception as e2:
                logger.debug("StatePredictor structured extraction failed: %s", e2)
            logger.debug("StatePredictor fallback to raw content")
            return {'prediction': last_content or ''}
        except GraphRecursionError as e:
            logger.warning("StatePredictor hit recursion limit for %s", focal_class or '?')
            try:
                plain_llm = llm.model_copy(update={'streaming': False})
                result = plain_llm.invoke([
                    SystemMessage(content=system_prompt + "\n\nPredict program state from the provided analysis and test prefix only. Do NOT call any tools."),
                    HumanMessage(content=human_msg),
                ])
                raw = result.content if hasattr(result, 'content') else str(result)
                if raw:
                    return {'prediction': raw}
            except Exception as e2:
                if is_quota_error(e2):
                    raise
                logger.warning("StatePredictor fallback LLM call failed: %s", e2)
            return {'prediction': ''}
        except Exception as e:
            if is_quota_error(e):
                raise
            logger.warning("StatePredictor stopped early for %s: %s", focal_class or '?', e)
            return {'prediction': ''}

    return node

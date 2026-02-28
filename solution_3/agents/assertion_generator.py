import logging

from langchain_core.messages import SystemMessage, HumanMessage
from prompts.assertion_generator import ASSERTION_GENERATOR_SYSTEM, ASSERTION_GENERATOR_HUMAN
from schemas import AssertionOutput

logger = logging.getLogger(__name__)


def make_generator_node(llm):
    structured_llm = llm.with_structured_output(AssertionOutput)

    def node(state):
        messages = [
            SystemMessage(content=ASSERTION_GENERATOR_SYSTEM),
            HumanMessage(content=ASSERTION_GENERATOR_HUMAN.format(
                return_type=state.get('return_type', ''),
                prediction=state.get('prediction', ''),
                test_prefix=state['test_prefix'],
            )),
        ]
        try:
            result = structured_llm.invoke(messages)
            if isinstance(result, AssertionOutput):
                return {'assertion': result.assertion or ''}
        except Exception as e:
            logger.warning("Structured output failed, falling back to raw: %s", e)
        # Fallback: raw LLM call
        response = llm.invoke(messages)
        return {'assertion': response.content or ''}

    return node

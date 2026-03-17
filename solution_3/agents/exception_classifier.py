import logging

from langchain_core.messages import HumanMessage, SystemMessage
from prompts.exception_classifier import (
    EXCEPTION_CLASSIFIER_SYSTEM,
    EXCEPTION_CLASSIFIER_HUMAN,
)
from schemas import ExceptionClassification

logger = logging.getLogger(__name__)


def make_exception_classifier_node(llm):
    """Return a LangGraph node that classifies exception vs assertion."""
    structured_llm = llm.with_structured_output(ExceptionClassification)

    def exception_classifier_node(state: dict) -> dict:
        prompt = EXCEPTION_CLASSIFIER_HUMAN.format(
            focal_method=state.get("focal_method", ""),
            docstring=state.get("docstring", "") or "(no docstring)",
            test_prefix=state.get("test_prefix", ""),
        )
        messages = [
            SystemMessage(content=EXCEPTION_CLASSIFIER_SYSTEM),
            HumanMessage(content=prompt),
        ]
        try:
            result: ExceptionClassification = structured_llm.invoke(messages)
            return {
                "is_exception": result.is_exception,
                "exception_reasoning": result.reasoning,
            }
        except Exception as e:
            logger.warning("Exception classifier failed: %s", e)
            return {"is_exception": False, "exception_reasoning": ""}

    return exception_classifier_node

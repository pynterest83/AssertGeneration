import logging

from langchain_core.messages import HumanMessage, SystemMessage
from prompts.exception_classifier import (
    EXCEPTION_CLASSIFIER_SYSTEM,
    EXCEPTION_CLASSIFIER_HUMAN,
)
from schemas import ExceptionClassification
from lang_config import LANG_CONFIGS

logger = logging.getLogger(__name__)


def make_exception_classifier_node(llm):
    structured_llm = llm.with_structured_output(ExceptionClassification)

    def exception_classifier_node(state: dict) -> dict:
        lang_cfg = LANG_CONFIGS.get(state.get('language', 'java'), LANG_CONFIGS['java'])
        system_prompt = EXCEPTION_CLASSIFIER_SYSTEM.format(**lang_cfg)

        prompt = EXCEPTION_CLASSIFIER_HUMAN.format(
            focal_method=state.get("focal_method", ""),
            docstring=state.get("docstring", "") or "(no docstring)",
            test_prefix=state.get("test_prefix", ""),
            doc_keyword=lang_cfg['doc_keyword'],
            code_fence=lang_cfg['code_fence'],
        )
        messages = [
            SystemMessage(content=system_prompt),
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


# Heuristic fallback
#
# import re
#
# _EXCEPTION_PATTERNS = re.compile(
#     r'// Undeclared exception!'
#     r'|assertThrows|expectThrows'
#     r'|@Test\s*\(\s*expected\s*='
#     r'|try\s*\{.*?fail\s*\('
#     r'|shouldThrow\s*\(|assertRaises|pytest\.raises'
#     r'|expect\(\s*\w+\s*\)\.toThrow'
#     r'|assertThatExceptionOfType',
#     re.IGNORECASE | re.DOTALL,
# )
#
# def make_exception_classifier_node(llm):
#     def exception_classifier_node(state: dict) -> dict:
#         test_prefix = state.get("test_prefix", "")
#         is_exc = bool(_EXCEPTION_PATTERNS.search(test_prefix))
#         logger.debug("Exception classifier (heuristic): is_exception=%s", is_exc)
#         return {
#             "is_exception": is_exc,
#             "exception_reasoning": "heuristic" if is_exc else "",
#         }
#     return exception_classifier_node

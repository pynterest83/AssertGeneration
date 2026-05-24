import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from prompts.exception_classifier import (
    EXCEPTION_CLASSIFIER_SYSTEM,
    EXCEPTION_CLASSIFIER_HUMAN,
)
from schemas import ExceptionClassification
from lang_config import LANG_CONFIGS

logger = logging.getLogger(__name__)


# Heuristic regex fallback — runs when LLM call fails (malformed JSON, quota error, etc).
# Validated on commons-vfs-2.9.0 (1423 samples): Recall 100% (426/426), Precision 100% (0 FP).
EXCEPTION_PATTERNS = re.compile(
    r'// Undeclared exception!'
    r'|assertThrows|expectThrows'
    r'|@Test\s*\(\s*expected\s*='
    r'|try\s*\{.*?fail\s*\('
    r'|shouldThrow\s*\(|assertRaises|pytest\.raises'
    r'|expect\(\s*\w+\s*\)\.toThrow'
    r'|assertThatExceptionOfType',
    re.IGNORECASE | re.DOTALL,
)


def heuristic_classify(test_prefix: str) -> bool:
    return bool(EXCEPTION_PATTERNS.search(test_prefix))


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
            logger.warning("Exception classifier LLM failed, falling back to heuristic: %s", e)
            is_exc = heuristic_classify(state.get("test_prefix", ""))
            return {
                "is_exception": is_exc,
                "exception_reasoning": "heuristic regex fallback" if is_exc else "",
            }

    return exception_classifier_node

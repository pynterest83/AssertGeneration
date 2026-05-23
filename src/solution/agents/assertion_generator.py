import logging

from langchain_core.messages import SystemMessage, HumanMessage
from prompts.assertion_generator import ASSERTION_GENERATOR_HUMAN, ASSERTION_GENERATOR_SYSTEM_MAP
from schemas import AssertionOutput
from lang_config import LANG_CONFIGS
from tools.definitions import is_quota_error

logger = logging.getLogger(__name__)


def make_generator_node(llm):
    # streaming=False required — streaming=True causes hangs on structured output calls.
    structured_llm = llm.model_copy(update={'streaming': False}).with_structured_output(AssertionOutput)

    def node(state):
        lang = state.get('language', 'java').lower()
        lang_cfg = LANG_CONFIGS.get(lang, LANG_CONFIGS['java'])
        system_template = ASSERTION_GENERATOR_SYSTEM_MAP.get(lang, ASSERTION_GENERATOR_SYSTEM_MAP['java'])
        system_prompt = system_template.format(**lang_cfg)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=ASSERTION_GENERATOR_HUMAN.format(
                return_type=state.get('return_type', ''),
                prediction=state.get('prediction', ''),
                test_prefix=state['test_prefix'],
            )),
        ]
        assertion = ''
        try:
            result = structured_llm.invoke(messages)
            assertion = result.assertion or ''
        except Exception as e:
            if is_quota_error(e):
                raise
            logger.warning("Structured output failed, falling back to raw: %s", e)
            try:
                raw = llm.invoke(messages)
                assertion = raw.content or ''
            except Exception as e2:
                if is_quota_error(e2):
                    raise
                logger.warning("Raw fallback also failed: %s", e2)
                assertion = ''

        if lang == 'python':
            assertion = assertion.strip().rstrip(';')

        return {'assertion': assertion}

    return node

from typing import TypedDict

MAX_AGENT_STEPS = 15

class AssertionState(TypedDict):
    focal_method: str
    focal_class: str
    language: str       # "java", "python", "javascript"
    docstring: str
    test_prefix: str
    return_type: str
    test_name: str
    file_path: str

    is_exception: bool
    exception_reasoning: str

    analysis: str
    prediction: str
    assertion: str

    # passed to state_predictor to pre-seed its ext_cache and avoid re-query.
    known_external: list

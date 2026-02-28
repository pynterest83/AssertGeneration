from typing import TypedDict

MAX_AGENT_STEPS = 50


class AssertionState(TypedDict):
    focal_method: str
    focal_class: str
    docstring: str
    test_prefix: str
    return_type: str
    test_name: str
    file_path: str

    analysis: str
    prediction: str
    assertion: str

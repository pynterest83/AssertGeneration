"""Per-language assertion injectors. Dispatches by language string."""
from typing import Callable


def inject_tests(language: str, repo_dir: str, preds_csv: str) -> list[str]:
    """Inject predicted assertions into test files for the given language.

    Returns a list of absolute paths of modified files.
    """
    fn: Callable[[str, str], list[str]]
    if language == "python":
        from .python_injector import inject_tests as fn
    elif language == "java":
        from .java_injector import inject_tests as fn
    else:
        return []
    return fn(repo_dir, preds_csv)

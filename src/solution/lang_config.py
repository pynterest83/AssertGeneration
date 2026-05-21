LANG_CONFIGS = {
    "java": {
        "language": "Java",
        "test_framework": "JUnit 4",
        "assertion_methods": (
            "assertEquals, assertTrue, assertFalse, assertNull, "
            "assertNotNull, assertSame, assertNotSame"
        ),
        "doc_keyword": "javadoc",
        "code_fence": "java",
        "file_extensions": [".java"],
    },
    "python": {
        "language": "Python",
        "test_framework": "pytest",
        "assertion_methods": (
            "assert x == y, assert x is None, assert x is True, "
            "assert x is False, assert x in collection"
        ),
        "doc_keyword": "docstring",
        "code_fence": "python",
        "file_extensions": [".py"],
    },
    "javascript": {
        "language": "JavaScript",
        "test_framework": "Jest",
        "assertion_methods": (
            "expect(x).toBe(y), expect(x).toEqual(y), expect(x).toBeNull(), "
            "expect(x).toBeTruthy(), expect(x).toBeFalsy(), expect(x).toHaveLength(n)"
        ),
        "doc_keyword": "JSDoc",
        "code_fence": "javascript",
        "file_extensions": [".js", ".jsx"],
    },
}

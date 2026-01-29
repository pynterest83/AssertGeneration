def build_prompt_with_instruction(
    focal_method: str,
    test_prefix: str,
    docstring: str = "",
    context: str = "",
    oracle_type: str = "assertion"
) -> str:
    prompt_parts = []
    instruction = """Write the assertion for this JUnit 4 test.
Output ONLY the assertion, nothing else.
YOUR ANSWER MUST NOT BE EMPTY.

Valid JUnit 4 assertions (use ONLY these):
- assertEquals(expected, actual)
- assertTrue(condition)
- assertFalse(condition)
- assertNull(object)
- assertNotNull(object)
- assertSame(expected, actual)
- assertNotSame(expected, actual)

DO NOT use assertThrows (JUnit 5 only).
For negative numbers, use parentheses: assertEquals((-1), result);

Example: assertEquals(0, result);"""
    
    prompt_parts.append(instruction)
    
    if focal_method and focal_method.strip() and "nan" not in focal_method.lower():
        prompt_parts.append(f"Method:\n{focal_method}")
    
    if docstring and docstring.strip() and "nan" not in docstring.lower():
        prompt_parts.append(f"Doc: {docstring}")
    
    if context and context.strip():
        prompt_parts.append(f"Context:\n{context}")
    
    prompt_parts.append(f"Test:\n{test_prefix}")
    
    prompt_parts.append("Assertion:")
    
    return "\n\n".join(prompt_parts)


def build_prompt_assertion_only(
    focal_method: str,
    test_prefix: str,
    docstring: str = "",
    context: str = ""
) -> str:
    return build_prompt_with_instruction(focal_method, test_prefix, docstring, context)

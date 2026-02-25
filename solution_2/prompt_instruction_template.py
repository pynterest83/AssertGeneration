def build_prompt_with_instruction(
    focal_method: str,
    test_prefix: str,
    docstring: str = "",
    context: str = "",
    return_type: str = "",
    oracle_type: str = "assertion"
) -> str:
    prompt_parts = []
    instruction = """Write the assertion for this JUnit 4 test.
Output ONLY the assertion, nothing else.
YOUR ANSWER MUST NOT BE EMPTY.

Valid JUnit 4 assertions:
- assertEquals(expected, actual)
- assertTrue(condition)
- assertFalse(condition)
- assertNull(object)
- assertNotNull(object)
- assertSame(expected, actual)
- assertNotSame(expected, actual)

IMPORTANT - Choose assertion based on return type:
- void → no return value, use assertNull/assertNotNull on side effects
- boolean → use assertTrue/assertFalse
- int/long/double/float → use assertEquals with numeric value
- String → use assertEquals with string value
- Object/Collection/Array → use assertNotNull, assertEquals, or assertSame

DO NOT use assertThrows (JUnit 5 only).
For negative numbers: assertEquals((-1), result);"""
    
    prompt_parts.append(instruction)
    
    if return_type and return_type.strip():
        prompt_parts.append(f"Return Type: {return_type}")
    
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

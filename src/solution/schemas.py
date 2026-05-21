from pydantic import BaseModel, Field, field_validator


class ExceptionClassification(BaseModel):
    """Structured output from the exception classifier agent."""

    is_exception: bool = Field(
        description="True if the test expects an exception, False if it needs an assertion."
    )
    reasoning: str = Field(
        description="One-sentence explanation of the classification."
    )


class CodeAnalysis(BaseModel):
    """Structured output from the code analyzer agent."""

    signature: str = Field(
        description="Full method signature including return type, name, and parameters"
    )
    fields_summary: str = Field(
        default="",
        description="Summary of class fields, their types, visibility, and how they are initialized (e.g. via constructor)"
    )
    branches: list[str] = Field(
        default_factory=list,
        description="List of logic branches (if/else, switch, try/catch) with conditions",
    )
    return_conditions: list[str] = Field(
        default_factory=list,
        description="List of return conditions: what value is returned under which condition",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of external method calls or types this method depends on",
    )

    @field_validator('fields_summary', mode='before')
    @classmethod
    def _str_or_empty(cls, v):
        return v if v is not None else ''

    @field_validator('branches', 'return_conditions', 'dependencies', mode='before')
    @classmethod
    def _list_or_empty(cls, v):
        return v if v is not None else []


class StatePrediction(BaseModel):
    """Structured output from the state predictor agent."""

    variable_states: list[str] = Field(
        default_factory=list,
        description="List of variable states after test prefix execution, e.g. 'int0 = -2', 'string0 = null'",
    )
    observable_state: str = Field(
        description="The concrete observable state of the last assigned variable or return value that can be asserted on (using public getters or the variable itself)"
    )
    inaccessible: list[str] = Field(
        default_factory=list,
        description="List of fields/state that are private or inaccessible from outside the class (no public getter)",
    )
    assertion_target: str = Field(
        description="The recommended variable or expression to assert on, e.g. 'int0', 'string0.length()', 'list0.size()'"
    )

    @field_validator('observable_state', 'assertion_target', mode='before')
    @classmethod
    def _str_or_empty(cls, v):
        return v if v is not None else ''

    @field_validator('variable_states', 'inaccessible', mode='before')
    @classmethod
    def _list_or_empty(cls, v):
        return v if v is not None else []


class AssertionOutput(BaseModel):
    """Structured output from the assertion generator."""

    assertion: str = Field(
        description="A single test assertion statement in the target language and framework. Do not add semicolons unless required by the language."
    )

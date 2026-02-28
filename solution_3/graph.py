from langgraph.graph import StateGraph, START, END
from state import AssertionState
from agents.code_analyzer import make_analyzer_node
from agents.state_predictor import make_predictor_node
from agents.assertion_generator import make_generator_node
from tools.definitions import create_tools


def build_graph(llm, method_store):
    tools = create_tools(method_store)

    graph = StateGraph(AssertionState)
    graph.add_node("code_analyzer", make_analyzer_node(llm, tools))
    graph.add_node("state_predictor", make_predictor_node(llm, tools))
    graph.add_node("assertion_generator", make_generator_node(llm))

    graph.add_edge(START, "code_analyzer")
    graph.add_edge("code_analyzer", "state_predictor")
    graph.add_edge("state_predictor", "assertion_generator")
    graph.add_edge("assertion_generator", END)

    return graph.compile()

from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from langgraph.graph.graph import START, END

from app.graph.utils import pipe_subgraph

from .nodes.design_chapter import design_chapter
from .nodes.generate_html import generate_html
from .nodes.generate_content import generate_content
from .nodes.generate_gliastar import generate_gliastar
from .nodes.collect_ressult import collect_ressult
from .nodes.get_gliastar_url import get_gliastar_url
from .schema import State, SubState


def build() -> Pregel:
    workflow = StateGraph(State)
    workflow.add_node("design_chapter", design_chapter)
    workflow.add_node("generate_html", generate_html)
    workflow.add_node("generate_content", generate_content)
    workflow.add_node("generate_gliastar", generate_gliastar)
    workflow.add_node("collect_ressult", collect_ressult)
    workflow.add_node("get_gliastar_url", get_gliastar_url)

    workflow.add_edge(START, "design_chapter")
    workflow.add_edge("design_chapter", "generate_html")
    workflow.add_edge("design_chapter", "generate_content")
    workflow.add_edge("design_chapter", "generate_gliastar")
    workflow.add_edge("generate_html", "collect_ressult")
    workflow.add_edge("generate_content", "collect_ressult")
    workflow.add_edge("generate_gliastar", "collect_ressult")
    workflow.add_edge("collect_ressult", "get_gliastar_url")
    workflow.add_edge("get_gliastar_url", END)

    app = workflow.compile()
    return pipe_subgraph(app, SubState)

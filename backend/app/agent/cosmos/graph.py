from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from langgraph.graph.graph import START, END

from app.graph.utils import pipe_subgraph

from .nodes.design_chapter import design_chapter
from .nodes.generate_cosmos_video import generate_cosmos_video
from .nodes.collect_result import collect_result
from .nodes.get_cosmos_url import get_cosmos_url
from .schema import State, SubState


def build() -> Pregel:
    workflow = StateGraph(State)
    workflow.add_node("design_chapter", design_chapter)
    workflow.add_node("generate_cosmos_video", generate_cosmos_video)
    workflow.add_node("collect_result", collect_result)
    workflow.add_node("get_cosmos_url", get_cosmos_url)

    workflow.add_edge(START, "design_chapter")
    workflow.add_edge("design_chapter", "generate_cosmos_video")
    workflow.add_edge("generate_cosmos_video", "collect_result")
    workflow.add_edge("collect_result", "get_cosmos_url")
    workflow.add_edge("get_cosmos_url", END)

    app = workflow.compile()
    return pipe_subgraph(app, SubState)

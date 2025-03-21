from langgraph.pregel import Pregel

from .schema import GraphType
from .threejs.graph import build as threejs_graph


GRAPH: dict[GraphType, Pregel] = {
    GraphType.threejs: threejs_graph(),
}

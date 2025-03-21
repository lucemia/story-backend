from pydantic import BaseModel
from enum import Enum

from langchain_core.runnables.config import RunnableConfig
from app.llms import AgentType


class GraphType(str, Enum):
    writer = "writer"
    tool = "tool"
    threejs = "threejs"
    canvas_debug = "canvas_debug"


class AgentConfig(BaseModel):
    system_message: str = "You are a helpful assistant"
    agent_type: AgentType = AgentType.GPT_4O
    graph_type: GraphType = GraphType.tool
    interrupt_before_action: bool = False
    assistant_id: str | None
    thread_id: str | None
    user_id: str | None


class NamedAgentConfig(AgentConfig):
    name: str = "Helpful Agent"
    expert_configs: list[RunnableConfig] = []

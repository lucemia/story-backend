from pydantic import BaseModel
from typing import Any, Annotated
from enum import Enum
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
)

from app.llms import AgentType


SUPERVISOR = "regular"


class Language(Enum):
    zh_tw = "traditional Chinese"
    eng = "English"


class ModelState(BaseModel):
    def to_dict(self, exclude_defaults: bool = True):
        changed_fields = self.__fields_set__ if exclude_defaults else self.dict()
        return {key: getattr(self, key) for key in changed_fields}


class StateBase(ModelState):
    messages: Annotated[list[BaseMessage], add_messages] = []
    hidden_messages: Annotated[list[BaseMessage], add_messages] = []
    language: Language = Language.zh_tw
    assistant_id: str = SUPERVISOR
    agent_type: AgentType | None
    system_prompt: str | None
    canvas: str = ""


class AppState(StateBase):
    substate: ModelState | None


class LiberalToolMessage(ToolMessage):
    content: Any


class RouterMessage(AIMessage):
    type: str = "router"
    assistant_id: str = ""
    expert: str


class Article(BaseModel):
    title: str
    content: str

    def to_str(self):
        return f"## {self.title}\n{self.content}"


class Scene(BaseModel):
    html: str
    gliastar: str
    article: Article

    def to_str(self):
        return (
            f"{self.article.to_str()}\n\nhtml: {self.html}\ngliastar: {self.gliastar}"
        )


class HackathonResultMessage(AIMessage):
    type: str = "result"
    scenes: list[Scene]

from typing import Callable, Any, Awaitable, Type
from functools import wraps
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    FunctionMessage,
    HumanMessage,
)
from langgraph.pregel import Pregel
from langchain_core.runnables.config import RunnableConfig

from app.schema import Assistant
from app.graph.schema import SUPERVISOR

from .schema import (
    LiberalToolMessage,
    StateBase as State,
    RouterMessage,
    AppState,
    ModelState,
)


def parse_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    msgs = []
    for m in messages:
        if isinstance(m, LiberalToolMessage):
            _dict = m.dict()
            _dict["content"] = str(_dict["content"])
            m_c = ToolMessage(**_dict)
            msgs.append(m_c)
        elif isinstance(m, FunctionMessage):
            msgs.append(HumanMessage(content=str(m.content)))
        elif isinstance(m, RouterMessage):
            msgs = msgs[:-1] if m.expert != SUPERVISOR else msgs
        else:
            msgs.append(m)

    return msgs


def get_configurable(
    assistant: Assistant, thread_id: str, user_id: str
) -> dict[str, str]:
    return {
        **assistant["config"]["configurable"],
        "name": assistant["name"],
        "thread_id": thread_id,
        "user_id": user_id,
        "assistant_id": assistant["assistant_id"],
    }


def find_assistant_config(
    expert_configs: list[RunnableConfig], assistant_id: str
) -> dict[str, Any]:
    expert_config = next(
        (
            config["configurable"]
            for config in expert_configs
            if config["configurable"]["assistant_id"] == assistant_id
        ),
        None,
    )
    assert expert_config, ValueError(
        f"No expert config with assistant id {assistant_id} found"
    )

    return expert_config


def as_edge(func: Callable[[State], str]) -> Callable[[str], Callable[[State], str]]:
    def wrapper(edge_name: str) -> Callable[[State], str]:
        func.__name__ = edge_name
        return func

    return wrapper


def as_node(
    func: Callable[[State], Awaitable[State]]
) -> Callable[[State], dict[str, Any]]:
    @wraps(func)
    async def node_function(state: State) -> dict[str, Any]:
        result = await func(state)
        return result.to_dict()

    return node_function


def pipe_subgraph(app: Pregel, SubState: Type[ModelState]):
    def pre(state: AppState) -> dict[str, Any]:
        substate = state.substate or SubState()
        substate_dict = substate.to_dict(exclude_defaults=False)
        return {**substate_dict, **state.to_dict(exclude_defaults=False)}

    def post(state_dict: dict[str, Any]) -> AppState:
        state = SubState(**state_dict)
        return AppState(**{**state_dict, "substate": state})

    return pre | app | post

import pickle
from typing import Any, Callable, ParamSpec, TypeVar
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint import CheckpointAt
from langgraph.graph import StateGraph
from langgraph.graph.graph import START, END
from langgraph.pregel import Pregel

from app.checkpoint import PostgresCheckpoint
from app.graph.schema import AppState

from .schema import NamedAgentConfig, AgentConfig
from .utils import (
    dump_graph_png,
    init_state,
    get_agent,
    get_clean_state,
)

P = ParamSpec("P")
R = TypeVar("R")


def bind_param_spec(_: Callable[P, Any]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    return lambda func: func


CHECKPOINTER = PostgresCheckpoint(serde=pickle, at=CheckpointAt.END_OF_STEP)
DEBUG = False


@bind_param_spec(NamedAgentConfig)
def build_agent(*args: Any, **kwargs: Any) -> Pregel:
    workflow = StateGraph(AppState)
    config = NamedAgentConfig(*args, **kwargs)
    agent_name = config.name

    workflow.add_node(agent_name, get_agent(config))
    workflow.add_node("clean_state", get_clean_state(config))
    workflow.add_edge(START, agent_name)
    workflow.add_edge(agent_name, "clean_state")
    workflow.add_edge("clean_state", END)

    app = workflow.compile(
        checkpointer=CHECKPOINTER,
    )

    if DEBUG:
        print("graph:", dump_graph_png(app))

    return app.with_config(
        RunnableConfig(
            recursion_limit=50,
            configurable={
                "assistant_id": config.assistant_id,
                "thread_id": config.thread_id,
                "user_id": config.user_id,
            },
        )
    )


def state_init_graph(config: AgentConfig) -> Pregel:
    workflow = StateGraph(AppState)
    workflow.add_node("init_state", init_state)
    workflow.add_edge("init_state", END)
    workflow.set_entry_point("init_state")
    return workflow.compile(checkpointer=CHECKPOINTER).with_config(
        RunnableConfig(
            recursion_limit=50,
            configurable={
                "assistant_id": config.assistant_id,
                "thread_id": config.thread_id,
                "user_id": config.user_id,
            },
        )
    )

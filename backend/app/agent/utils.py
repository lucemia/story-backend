import tempfile

from langgraph.graph.graph import CompiledGraph

from app.graph.schema import AppState, SUPERVISOR
from app.graph.utils import as_node, find_assistant_config
from app.graph.schema import RouterMessage

from .graph import GRAPH
from .schema import NamedAgentConfig


def dump_graph_png(app: CompiledGraph) -> str:
    graph_img = app.get_graph().draw_mermaid_png()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as fp:
        fp.write(graph_img)
    return fp.name


def find_named_agent_config(config: NamedAgentConfig, assistant_id: str):
    if assistant_id != SUPERVISOR:
        assistant_config_dict = find_assistant_config(
            config.expert_configs, assistant_id
        )
        return NamedAgentConfig(**assistant_config_dict)
    return config


def get_agent(config: NamedAgentConfig):
    @as_node
    async def agent(state: AppState) -> AppState:
        assistant_config = find_named_agent_config(config, state.assistant_id)
        return await GRAPH[assistant_config.graph_type].ainvoke(state)

    return agent


def get_clean_state(config: NamedAgentConfig):
    @as_node
    async def clean_state(state: AppState) -> AppState:
        return AppState(
            messages=[RouterMessage(content=SUPERVISOR, expert=SUPERVISOR)],
            assistant_id=SUPERVISOR,
            agent_type=config.agent_type,
            system_prompt=config.system_message,
        )

    return clean_state


@as_node
async def init_state(state: AppState) -> AppState:
    return state

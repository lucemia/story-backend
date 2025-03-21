from uuid import uuid4

from fastapi import APIRouter
from sse_starlette import EventSourceResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

import app.storage as storage
from app.stream import astream_state, to_sse
from app.agent.build import state_init_graph, build_agent
from app.agent.schema import AgentConfig
from app.graph.schema import StateBase as State
from app.graph.utils import get_configurable, parse_messages

from .dummy import dummy_sse
from .utils import run_input_and_config, parse_experts_from_tools
from .schema import HeadlessPayload


router = APIRouter()
user_id = "a66a45b8-d08c-4ab1-abfb-d4fd4c7356da"
assistant_id = "acc846ce-9a9f-4c04-8e13-7498e12696b8"


async def create_thread() -> str:
    thread = await storage.put_thread(
        user_id,
        str(uuid4()),
        assistant_id=assistant_id,
        name="threejs_demo",
    )
    assistant = await storage.get_assistant(user_id, assistant_id)
    configurable = get_configurable(assistant, thread["thread_id"], user_id)
    config = AgentConfig(**configurable)
    init_graph = state_init_graph(config)
    await init_graph.ainvoke(
        State(
            system_prompt=config.system_message,
            agent_type=config.agent_type,
        ).to_dict(exclude_defaults=False),
        RunnableConfig(configurable=config.dict()),
    )

    return thread["thread_id"]


@router.post("/invoke")
async def invoke(payload: HeadlessPayload) -> str:
    thread_id = await create_thread()
    config = await run_input_and_config(thread_id, user_id)
    expert_configs = await parse_experts_from_tools(
        thread_id, user_id, config["configurable"]["tools"]
    )
    agent = build_agent(**config["configurable"], expert_configs=expert_configs)

    input_dict = State(
        messages=[HumanMessage(content=payload.message)],
    ).to_dict()

    response = await agent.ainvoke(input_dict, config)
    return parse_messages(State(**response).messages)[-1].content


@router.post("/stream")
async def stream_run(payload: HeadlessPayload):
    thread_id = await create_thread()
    config = await run_input_and_config(thread_id, user_id)
    expert_configs = await parse_experts_from_tools(
        str(thread_id), user_id, config["configurable"]["tools"]
    )
    agent = build_agent(**config["configurable"], expert_configs=expert_configs)

    input_dict = State(
        messages=[HumanMessage(content=payload.message)],
    ).to_dict()

    return EventSourceResponse(to_sse(astream_state(agent, input_dict, config)))


@router.post("/dummy_stream")
async def dummy_stream(payload: HeadlessPayload):
    return EventSourceResponse(dummy_sse())

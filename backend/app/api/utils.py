import asyncio
from typing import Any

from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig

from app.tools.base import ToolkitType
from app.storage import get_assistant, get_thread
from app.graph.utils import get_configurable


async def get_assistant_config(
    thread_id: str, user_id: str, assistant_id: str
) -> RunnableConfig:
    assistant = await get_assistant(user_id, assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    config: RunnableConfig = {
        **assistant["config"],
        "configurable": get_configurable(assistant, thread_id, user_id),
    }

    return config


async def run_input_and_config(thread_id: str, user_id: str):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    config = await get_assistant_config(str(thread_id), user_id, thread["assistant_id"])
    return config


async def parse_experts_from_tools(
    thread_id: str, user_id: str, tools: list[dict[str, Any]]
) -> list[RunnableConfig]:
    return await asyncio.gather(
        *(
            get_assistant_config(thread_id, user_id, tool["assistant_id"])
            for tool in tools
            if tool["type"] == ToolkitType.EXPERT
        )
    )

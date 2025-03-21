from functools import lru_cache
from enum import Enum
import os
import structlog
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = structlog.get_logger(__name__)

EMPTY_CALLBACKS_CONFIG = RunnableConfig(
    callbacks=BaseCallbackManager([]),
)


class AgentType(str, Enum):
    CLAUDE3 = "Claude 3"
    NVIDIA = "NVIDIA"


def convert_system_messages_to_human(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [
        HumanMessage(content=message.content)
        if isinstance(message, SystemMessage)
        else message
        for message in messages
    ]


@lru_cache(maxsize=1)
def get_claude():
    return ChatAnthropicVertex(
        model_name="claude-3-5-sonnet@20240620",
        project="living-bio",
        location="us-east5",
        temperature=1,
        max_output_tokens=8192,
    )


@lru_cache(maxsize=1)
def get_nvidia():
    return ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        temperature=0.1,
        api_key=os.environ["NVIDIA_API_KEY"],
        max_tokens=4096,
    )


def get_llm(agent: AgentType) -> BaseChatModel:
    match agent:
        case AgentType.CLAUDE3:
            return get_claude()
        case AgentType.NVIDIA:
            return get_nvidia()

    raise ValueError("Unexpected agent type")

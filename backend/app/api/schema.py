from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Sequence, Union

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig


class CreateRunPayload(BaseModel):
    """Payload for creating a run."""

    thread_id: str
    input: Optional[Union[Sequence[AnyMessage], Dict[str, Any]]] = Field(
        default_factory=dict
    )
    config: Optional[RunnableConfig] = None


class HeadlessPayload(BaseModel):
    message: str

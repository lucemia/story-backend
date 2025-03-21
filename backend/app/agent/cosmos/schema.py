from pydantic import BaseModel
from typing import Optional, List
from gq.pipe import AsyncResult

from app.graph.schema import StateBase, ModelState, Article


class SubState(ModelState):
    articles: list[Article] = []
    cosmos_videos: list[AsyncResult] = []


class State(StateBase, SubState):
    pass

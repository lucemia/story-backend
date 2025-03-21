from pydantic import BaseModel
from gq.pipe import AsyncResult

from app.graph.schema import StateBase, ModelState, Article


class ArticleList(BaseModel):
    articles: list[Article]


class SubState(ModelState):
    htmls: list[str] = []
    gliastars: list[AsyncResult] = []
    articles: list[Article] = []


class State(StateBase, SubState):
    pass

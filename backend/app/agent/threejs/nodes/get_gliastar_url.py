from tenacity import retry, stop_after_attempt, wait_fixed

from app.graph.utils import as_node
from app.graph.schema import HackathonResultMessage, Scene
from gq.pipe import AsyncResult

from ..schema import State


@as_node
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def get_gliastar_url(state: State) -> State:
    gliastar_urls: list[str] = [
        task.get_result(timeout=300)["file"]
        for task in state.gliastars
        if isinstance(task, AsyncResult)
    ]
    scenes = [
        Scene(html=html, gliastar=gliastar, article=article)
        for html, gliastar, article in zip(state.htmls, gliastar_urls, state.articles)
    ]
    return State(
        messages=[
            HackathonResultMessage(
                content="\n".join([f"- gliastar: {url}" for url in gliastar_urls]),
                scenes=scenes,
            )
        ]
    )

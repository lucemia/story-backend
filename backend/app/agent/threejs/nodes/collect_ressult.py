from app.graph.utils import as_node
from app.graph.schema import HackathonResultMessage, Scene

from ..schema import State


@as_node
async def collect_ressult(state: State) -> State:
    scenes = [
        Scene(html=html, gliastar=str(gliastar.key), article=article)
        for html, gliastar, article in zip(state.htmls, state.gliastars, state.articles)
    ]
    print("result collected")
    return State(
        messages=[
            HackathonResultMessage(
                content="\n".join([scene.to_str() for scene in scenes]), scenes=scenes
            )
        ]
    )

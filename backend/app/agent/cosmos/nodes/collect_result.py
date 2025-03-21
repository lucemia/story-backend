from app.graph.utils import as_node
from app.graph.schema import HackathonResultMessage, Scene

from ..schema import State


@as_node
async def collect_result(state: State) -> State:
    """
    Collect the results from the Cosmos video generation process.
    """
    # Create scenes with the Cosmos video keys
    scenes = [
        Scene(
            cosmos_video=str(cosmos_video.key),
            article=article
        )
        for cosmos_video, article in zip(state.cosmos_videos, state.articles)
    ]
    
    print("Cosmos results collected")
    return State(
        messages=[
            HackathonResultMessage(
                content="\n".join([f"- Cosmos video: {scene.cosmos_video}" for scene in scenes]),
                scenes=scenes
            )
        ]
    )

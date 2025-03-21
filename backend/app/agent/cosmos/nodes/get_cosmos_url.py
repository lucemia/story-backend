from tenacity import retry, stop_after_attempt, wait_fixed

from app.graph.utils import as_node
from app.graph.schema import HackathonResultMessage, Scene
from gq.pipe import AsyncResult

from ..schema import State


@as_node
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def get_cosmos_url(state: State) -> State:
    """
    Get the URLs for the generated Cosmos videos.
    """
    # Get the results from the AsyncResult objects
    cosmos_urls = []
    for task in state.cosmos_videos:
        if isinstance(task, AsyncResult):
            try:
                result = task.get_result(timeout=300)
                video_url = result.get("file", "")
                cosmos_urls.append(video_url)
            except Exception as e:
                print(f"Error getting Cosmos video URL: {e}")
                cosmos_urls.append("")
    
    # Create scenes with the video URLs
    scenes = [
        Scene(
            cosmos_video=cosmos_url,
            article=article
        )
        for cosmos_url, article in zip(cosmos_urls, state.articles)
    ]
    
    return State(
        messages=[
            HackathonResultMessage(
                content="\n".join([f"- Cosmos video: {url}" for url in cosmos_urls]),
                scenes=scenes,
            )
        ]
    )

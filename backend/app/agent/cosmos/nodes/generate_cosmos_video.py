import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed

from app.graph.utils import as_node
from app.llms import get_llm, get_nvidia

from ..schema import State, Article
from ..cosmos_video import create_cosmos_video


@as_node
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def generate_cosmos_video(state: State) -> State:
    """
    Generate videos using Nvidia's Cosmos API based on the scene descriptions.
    """
    try:
        # Generate videos for each article
        cosmos_videos = []
        for article in state.articles:
            # Use the article content as the prompt for Cosmos
            cosmos_result = create_cosmos_video(article.content)
            cosmos_videos.append(cosmos_result)
            
        return State(cosmos_videos=cosmos_videos)
    except Exception as e:
        print(f"Error generating Cosmos videos: {e}")
        # Return empty state in case of error
        return State(cosmos_videos=[])

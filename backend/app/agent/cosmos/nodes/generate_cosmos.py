import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_fixed

from app.graph.utils import as_node
from app.llms import get_llm, EMPTY_CALLBACKS_CONFIG, get_nvidia

from ..schema import State, Article
from ..cosmos_video import create_cosmos_video
from ..prompt import COSMOS_SCENE_PROMPT


async def generate_cosmos_videos(llm: BaseChatModel, articles: list[Article]):
    """Generate Cosmos videos for each article."""
    response = await asyncio.gather(
        *(
            llm.ainvoke(
                [
                    SystemMessage(content=COSMOS_SCENE_PROMPT.format(topic=article.to_str())),
                    HumanMessage(content=f"Create a detailed scene description for this topic: {article.to_str()}"),
                ],
                EMPTY_CALLBACKS_CONFIG,
            )
            for article in articles
        )
    )
    return [create_cosmos_video(r.content) for r in response]


@as_node
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def generate_cosmos(state: State) -> State:
    """Generate Cosmos videos for each article in the state."""
    try:
        cosmos_videos = await generate_cosmos_videos(get_nvidia(), state.articles)
    except Exception as e:
        print(f"Nvidia might have failed: {e}")
        cosmos_videos = await generate_cosmos_videos(
            get_llm(state.agent_type), state.articles
        )
    return State(cosmos_videos=cosmos_videos)

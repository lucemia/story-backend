import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_fixed

from app.graph.utils import as_node
from app.llms import get_llm, EMPTY_CALLBACKS_CONFIG, get_nvidia

from ..schema import State, Article
from ...gliastar import create_star_video
from ..prompt import CREATE_LINES_PROMPT


async def generate_speech_and_task(llm: BaseChatModel, articles: list[Article]):
    response = await asyncio.gather(
        *(
            llm.ainvoke(
                [
                    SystemMessage(content=CREATE_LINES_PROMPT),
                    HumanMessage(content=article.to_str()),
                ],
                EMPTY_CALLBACKS_CONFIG,
            )
            for article in articles
        )
    )
    return [create_star_video(r.content) for r in response]


@as_node
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def generate_gliastar(state: State) -> State:
    try:
        task_urls = await generate_speech_and_task(get_nvidia(), state.articles)
    except Exception as e:
        print(f"Nvidia might be failed : {e}")
        task_urls = await generate_speech_and_task(
            get_llm(state.agent_type), state.articles
        )
    return State(gliastars=task_urls)

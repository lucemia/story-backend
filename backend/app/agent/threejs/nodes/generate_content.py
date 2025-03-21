from langchain_core.messages import SystemMessage, HumanMessage
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed

from app.llms import get_llm, EMPTY_CALLBACKS_CONFIG
from app.graph.utils import as_node, parse_messages
from app.graph.schema import Article

from ..utils import last_human_message
from ..schema import State
from ..prompt import CREATE_CHAPTER_PROMPT


@as_node
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def generate_content(state: State) -> State:
    llm = get_llm(state.agent_type)
    history = parse_messages(state.messages)
    question = last_human_message(history)
    response = await asyncio.gather(
        *(
            llm.ainvoke(
                [
                    SystemMessage(content=CREATE_CHAPTER_PROMPT),
                    HumanMessage(content=article.to_str() + question.content),
                ],
                EMPTY_CALLBACKS_CONFIG,
            )
            for article in state.articles
        )
    )
    article_list = [
        Article(
            title=article.title,
            content=r.content,
        )
        for article, r in zip(state.articles, response)
    ]

    return State(articles=article_list)

from langchain_core.messages import SystemMessage

from app.llms import get_llm, get_openai_llm, EMPTY_CALLBACKS_CONFIG
from app.graph.utils import as_node, parse_messages

from ..utils import last_human_message
from ..schema import State, ArticleList
from ..prompt import CHAPTER_DESIGN_PROMPT


@as_node
async def design_chapter(state: State) -> State:
    llm = get_llm(state.agent_type)
    gpt = get_openai_llm()
    article_list_parser = gpt.with_structured_output(ArticleList)
    history = parse_messages(state.messages)
    response = await llm.ainvoke(
        [
            SystemMessage(content=CHAPTER_DESIGN_PROMPT),
            *history,
            last_human_message(history),
        ]
    )
    article_list: ArticleList = await article_list_parser.ainvoke(
        f"Please format each chapter into an article: {response.content}",
        EMPTY_CALLBACKS_CONFIG,
    )

    return State(messages=[response], articles=article_list.articles)

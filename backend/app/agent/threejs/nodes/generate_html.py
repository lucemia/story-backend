import re
import asyncio
import tempfile
from gfs.store import remote
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_fixed

from app.llms import get_llm, EMPTY_CALLBACKS_CONFIG
from app.graph.utils import as_node, parse_messages

from ..schema import State
from ..prompt import CREATE_HTML_PROMPT, SCENE_DESIGN_PROMPT


def extract_html_content(text: str) -> str:
    pattern = r"```html(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text


@as_node
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def generate_html(state: State) -> State:
    llm = get_llm(state.agent_type)
    history = parse_messages(state.messages)

    inspirations: list[AIMessage] = []
    for article in state.articles:
        inspiration = await llm.ainvoke(
            [
                SystemMessage(content=SCENE_DESIGN_PROMPT),
                HumanMessage(content=article.to_str()),
            ],
        )
        inspirations.append(inspiration)

    response = await asyncio.gather(
        *(
            llm.ainvoke(
                [
                    SystemMessage(content=CREATE_HTML_PROMPT),
                    *history,
                    HumanMessage(content=inspiration.content),
                ],
                EMPTY_CALLBACKS_CONFIG,
            )
            for inspiration in inspirations
        )
    )

    html_urls = []
    for r in response:
        with tempfile.NamedTemporaryFile(suffix=".html") as fp:
            fp.write(extract_html_content(r.content).encode("utf-8"))
            fp.flush()
            url = remote(fp.name)
            html_urls.append(url.as_source())

    return State(
        messages=inspirations,
        htmls=html_urls,
        canvas=extract_html_content(response[0].content),
    )

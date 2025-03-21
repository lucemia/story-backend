from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_fixed

from app.graph.utils import as_node
from app.llms import get_llm, EMPTY_CALLBACKS_CONFIG

from ..schema import State, Article


DESIGN_PROMPT = """
You are a creative director for a video production company.
Your task is to create a detailed description for a video scene based on the user's input.
The description will be used by Nvidia's Cosmos AI to generate a photorealistic video.

Guidelines:
1. Create a detailed, vivid scene description that Cosmos can visualize
2. Focus on visual elements, camera movements, lighting, and atmosphere
3. Avoid abstract concepts or non-visual elements
4. Keep the description concise but detailed (50-100 words)
5. Make sure the scene is realistic and can be rendered by an AI system

Format your response as a single paragraph with no additional text or explanations.
"""


@as_node
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def design_chapter(state: State) -> State:
    """
    Design chapters based on the user input to create detailed scene descriptions
    for Cosmos video generation.
    """
    llm = get_llm(state.agent_type)
    
    # Process each user message to create articles
    articles = []
    for message in state.messages:
        if message.type == "human":
            response = await llm.ainvoke(
                [
                    SystemMessage(content=DESIGN_PROMPT),
                    HumanMessage(content=message.content),
                ],
                EMPTY_CALLBACKS_CONFIG,
            )
            
            # Create an article with the scene description
            article = Article(
                title=f"Cosmos Scene: {message.content[:30]}...",
                content=response.content,
                source=message.content,
            )
            articles.append(article)
    
    return State(articles=articles)

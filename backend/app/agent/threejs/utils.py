from langchain_core.messages import HumanMessage, BaseMessage


def last_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    for message in messages[::-1]:
        if isinstance(message, HumanMessage):
            return message
    return None

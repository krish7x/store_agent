from typing_extensions import TypedDict
from typing import List
from langchain_core.messages import BaseMessage


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    chat_history: List[BaseMessage]
    selected_agent: str
    response: str
    cart: List
    store_code: str

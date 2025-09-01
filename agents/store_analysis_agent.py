from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from state.state import State


"""
Store Analysis Agent - WORK IN PROGRESS

This agent is currently under development and not yet implemented.
It will provide business insights, store performance analysis, and strategic recommendations
for CaratLane store owners.

TODO:
- Implement store performance metrics analysis
- Add product performance insights

Status: Not Implemented
"""

llm = get_llm()


prompt_txt = """
TO BE UPDATED
"""


def store_analysis_node(state: State):
    """Call the LLM with the current state."""
    query = state.get("query", "")

    messages = [
        SystemMessage(content=prompt_txt),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    return {"chat_history": [response], "response": "Store analysis is coming soon"}

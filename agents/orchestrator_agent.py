import logging
from typing import Literal, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import get_llm
from state.state import State
from agents.product_filter_agent import product_filter_node, query_executor_node
from agents.summary_agent import summary_node
from agents.store_analysis_agent import store_analysis_node
from helpers.redis_helper import get_simple_chat_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = get_llm()

ROUTER_PROMPT_SHORT = (
    "Classify the user query for routing. Reply with ONLY one token:\n"
    "- product_filter_node  → product-related search/filter queries such as 'jewellery_type,' 'metal,' 'purity,' 'relationship,' 'occasion,' 'price,' etc. that need SQL over products\n"
    "- store_analysis_node  → if it contains store in the query, store performance/ASP/trends/strategy/business insights which needs deep analysis of the store's data\n"
    "No extra words."
)


def orchestrator(state: State):
    logger.info("Orchestrator: Routing start")
    return {
        "chat_history": state.get("chat_history", []),
        "query": state.get("query", ""),
    }


def get_chat_history(user_email: str):
    return get_simple_chat_history(user_email)


def save_chat_history(user_email: str, chat_history: Sequence[BaseMessage]):
    history = get_chat_history(user_email)
    history.add_messages(chat_history)


def route_via_llm(
    state: State,
) -> Literal["product_filter_node", "store_analysis_node"]:
    query = (state.get("query") or "").strip()
    messages = [SystemMessage(content=ROUTER_PROMPT_SHORT), HumanMessage(content=query)]
    response = llm.invoke(messages)
    logger.info(f"Decision: {response.content}")
    decision = (response.content or "").strip().lower()
    return decision


def create_orchestrator_graph():
    builder = StateGraph(State)

    builder.add_node("orchestrator", orchestrator)
    builder.add_node("product_filter_node", product_filter_node)
    builder.add_node("query_executor_node", query_executor_node)
    builder.add_node("summary_node", summary_node)
    builder.add_node("store_analysis_node", store_analysis_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_via_llm,
        {
            "product_filter_node": "product_filter_node",
            "store_analysis_node": "store_analysis_node",
        },
    )

    builder.add_edge("product_filter_node", "query_executor_node")
    builder.add_edge("query_executor_node", "summary_node")
    builder.add_edge("summary_node", END)
    builder.add_edge("store_analysis_node", END)

    return builder.compile()


orchestrator_graph = create_orchestrator_graph()


def invoke_orchestrator(query: str, user_email: str) -> dict:
    try:
        # Re-enable Redis with proper error handling
        try:
            chat_history = get_chat_history(user_email)
            past_messages = chat_history.get_messages()
            logger.info(f"Retrieved {len(past_messages)} messages from Redis")

            # If we have too many messages (indicating potential issues), clear cache
            if len(past_messages) > 100:
                logger.warning("Too many messages in cache, clearing for fresh start")
                chat_history.clear()
                past_messages = []
        except Exception as redis_error:
            logger.warning(f"Redis error, using empty chat history: {redis_error}")
            past_messages = []

        initial_state = {
            "query": query,
            "chat_history": past_messages,
            "response": None,
            "cart": [],
            "store_code": None,
        }
        result = orchestrator_graph.invoke(initial_state)

        # Save to Redis with error handling
        try:
            save_chat_history(user_email, result.get("chat_history", []))
            logger.info(
                f"Saved {len(result.get('chat_history', []))} messages to Redis"
            )
        except Exception as redis_error:
            logger.warning(f"Failed to save to Redis: {redis_error}")

        logger.info("Orchestrator completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in orchestrator: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "chat_history": [], "query": query}

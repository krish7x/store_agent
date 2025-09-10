import logging
from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from state.state import State

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM
llm = get_llm()


def summary_node(state: State):
    """Generate a summary from the SQL query results."""
    try:
        chat_history = state.get("chat_history", [])
        query = state.get("query", "")

        # Create system prompt for summary generation
        system_prompt = """You are a helpful assistant that creates concise summaries of product search results.

        Based on the SQL query results, provide a clear summary that includes:
        - Number of products found
        - Key characteristics (price range, material, etc.)
        - Any relevant insights
        - Currency is INR

        Keep the summary concise and informative."""

        # Create messages for the LLM
        messages = [SystemMessage(content=system_prompt)]

        # Add the chat history to provide context, filtering out empty messages
        for msg in chat_history:
            if hasattr(msg, 'content') and msg.content and str(msg.content).strip():
                messages.append(msg)

        # Add the user query for context
        if query:
            messages.append(HumanMessage(content=f"User query: {query}"))

        # Generate summary
        response = llm.invoke(messages)

        # Append the response to chat history instead of replacing it
        updated_chat_history = chat_history + [response]

        logger.info("Summary generated successfully")

        return {"chat_history": updated_chat_history, "response": response}

    except Exception as e:
        logger.error(f"Error in summary generation: {e}")
        # Return the original chat history if there's an error
        return {"chat_history": chat_history, "response": None}

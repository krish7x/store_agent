from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from state import State
from langgraph.graph import StateGraph, START, END
from sql_assistant_agent import sql_assistant_graph as ProductFilterAgent

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


prompt_txt = """You are an orchestrator for CaratLane's inventory chatbot.

**Your Role**: Route user queries to the appropriate agent.

**Available Agent**: ProductFilterAgent (SQL Assistant) - handles all database queries and product analysis.

**Routing Rules**:
- All queries go to ProductFilterAgent
- This agent can handle: product searches, data analysis, SQL queries, statistics

**Examples**:
- "Show me rings" → ProductFilterAgent
- "How many products?" → ProductFilterAgent
- "What's the average price?" → ProductFilterAgent
- "Gold products under 50k" → ProductFilterAgent

**Response**: Always return "ProductFilterAgent"

Selected Agent:"""


def orchestrator(state: State) -> Command:
    """Route the user query to the ProductFilterAgent (SQL Assistant)."""
    query = state.get("query", "")

    # Create messages for the LLM - simplified for Gemini compatibility
    messages = [
        SystemMessage(content=prompt_txt),
        HumanMessage(content=query),
    ]

    # Get the LLM response
    response = llm.invoke(messages)  # noqa: F841

    # Always route to ProductFilterAgent (SQL Assistant)
    return Command(
        goto="ProductFilterAgent", update={"selected_agent": "ProductFilterAgent"}
    )


# Create the orchestrator graph
builder = StateGraph(State)

# Add nodes
builder.add_node("orchestrator", orchestrator)
builder.add_node("ProductFilterAgent", ProductFilterAgent)

# Add edges
builder.add_edge(START, "orchestrator")
builder.add_edge("ProductFilterAgent", END)

# Compile the graph
orchestrator_graph = builder.compile(name="OrchestratorAgent")


def invoke_orchestrator(
    query: str, chat_history: list = [], cart: list = [], store_code: str = ""
):
    """Invoke the orchestrator agent system."""
    invoke_data = {
        "query": query,
        "chat_history": chat_history,
        "cart": cart,
        "store_code": store_code,
    }
    return orchestrator_graph.invoke(invoke_data)


if __name__ == "__main__":
    # Test the orchestrator
    chat_history = [
        HumanMessage(content="Hi, I'm looking for some jewellery"),
        AIMessage(
            content="Hello! I'd be happy to help you find jewellery. What type are you looking for?"
        ),
    ]

    test_queries = [
        "Show me rings in rose gold",
        "How many products do we have in stock?",
        "What's the average price?",
        "I need diamond earrings under 50k",
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print(f"{'=' * 50}")
        result = invoke_orchestrator(query, chat_history)
        print(f"Selected Agent: {result.get('selected_agent', 'Unknown')}")
        print(f"Final State: {result}")

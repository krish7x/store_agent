import logging
from langgraph.graph import StateGraph, START, END
from agents.product_filter_agent import call_model, execute_tool
from agents.summary_agent import call_model as summary_call_model
from state.state import State

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def orchestrator(state: State):
    """Orchestrator node that routes the initial query to SQL generation."""
    logger.info("Orchestrator: Starting SQL generation")
    return {"chat_history": state["chat_history"], "query": state["query"]}


def sql_call_model(state: State):
    """Call the SQL model to generate SQL query."""
    logger.info("SQL Call Model: Generating SQL query")
    result = call_model(state)
    return result


def sql_execute_tool(state: State):
    """Execute the SQL tool with the generated query."""
    logger.info("SQL Execute Tool: Executing SQL query")
    result = execute_tool(state)
    return result


def summary_generation(state: State):
    """Generate summary from the SQL results."""
    logger.info("Summary Generation: Creating summary")
    result = summary_call_model(state)
    return result


# Create the graph
def create_orchestrator_graph():
    """Create the orchestrator graph with linear flow."""
    builder = StateGraph(State)

    # Add nodes
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("sql_call_model", sql_call_model)
    builder.add_node("sql_execute_tool", sql_execute_tool)
    builder.add_node("summary_generation", summary_generation)

    # Linear flow: START -> orchestrator -> sql_call_model -> sql_execute_tool -> summary_generation -> END
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "sql_call_model")
    builder.add_edge("sql_call_model", "sql_execute_tool")
    builder.add_edge("sql_execute_tool", "summary_generation")
    builder.add_edge("summary_generation", END)

    return builder.compile()


# Create the graph instance
orchestrator_graph = create_orchestrator_graph()


def invoke_orchestrator(query: str) -> dict:
    """Invoke the orchestrator with a user query."""
    try:
        # Initialize state
        initial_state = {
            "query": query,
            "chat_history": [],
            "response": None,
            "cart": [],
            "store_code": None,
        }

        # Run the graph
        result = orchestrator_graph.invoke(initial_state)
        logger.info("Orchestrator completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error in orchestrator: {e}")
        return {"error": str(e), "chat_history": [], "query": query}


if __name__ == "__main__":
    # Test the orchestrator
    test_query = "Show me rings in rose gold under 50k"
    result = invoke_orchestrator(test_query)
    print("Orchestrator Result:", result)

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

from llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from state import State

llm = get_llm()


prompt_txt = """
You are a Store Analysis Agent for an inventory operations chatbot for CartaLane: A Tata Product, a jewellery brand.

Your role is to provide business insights, store performance analysis, and strategic recommendations to help store owners make informed decisions about their inventory.

**CAPABILITIES:**
- Analyze store sales performance
- Provide insights on product performance
- Recommend     inventory strategies
- Analyze market trends and preferences
- Help with pricing strategies
- Provide business intelligence

**WHAT YOU CAN HELP WITH:**
- Store performance metrics
- Product performance analysis
- Sales trend analysis
- Inventory optimization recommendations
- Market preference insights
- Busine    ss strategy recommendations

**EXAMPLES OF QUERIES YOU HANDLE:**
- "What's my store's performance this month?"
- "Should I prefer white gold over rose gold?"
- "What's my Average Sel    ling Price (ASP)?"
- "Which products sell best in my store?"
- "How are my sales trending?"
- "What inventory should I focus on?"

**IMPORTANT:**
- Provide data-driven insights
- Be specific and actionable
- Consider the store's context
- Offer strategic recommendations
- Be helpful and professional

Remember: You're here to help store owners make better business decisions through data analysis and insights.
"""


def call_model(state: State):
    """Call the LLM with the current state."""
    query = state.get("query", "")

    messages = [
            SystemMessage(content=prompt_txt),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)

        return {"chat_history": [response], "response": response}


# Create the Store Analysis Agent graph
builder = StateGraph(State)

# Add nodes
builder.add_node("call_model", call_model)

# Add edges
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# Compile the graph
store_analysis_graph = builder.compile(name="StoreAnalysisAgent")


def invoke_store_analysis(query: str, chat_history: list = []):
    """Invoke the Store Analysis Agent."""
    invoke_data = {
        "query": query,
        "chat_history": chat_history,
    }
    return store_analysis_graph.invoke(invoke_data)


if __name__ == "__main__":
    # Test the Store Analysis Agent
    chat_history = [
        HumanMessage(content="Hi, I need help analyzing my store performance"),
        AIMessage(content="Hello! I'm your Store Analysis Agent. I can help you understand your store's performance and provide strategic insights. What would you like to analyze?"),
    ]

    test_queries = [
        "What's my store's performance this month?",
        "Should I prefer white gold over rose gold?",
        "What's my Average Selling Price?",
        "Which products sell best in my store?"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        result = invoke_store_analysis(query, chat_history)
        print(f"Result: {result}")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from state import State

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


prompt_txt = """You are a Summary Agent for CaratLane's inventory chatbot.

Your task is to create a concise, factual summary of database query results.

Guidelines:
- Keep it under 100 words
- Use simple, clean language without markdown formatting (no ** or * symbols)
- Present information as factual statements only
- No questions or suggestions
- Focus on the key numbers and results

Example:
"Found 4102 products matching your criteria. Price range: ₹50,000 to ₹60,000. Purity: 14 KT."

Be direct and factual."""


def call_model(state: State):
    """Call the LLM with the current state to generate a summary."""
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    agent_result_text = ""
    for msg in chat_history:
        if hasattr(msg, "content") and "Agent Result:" in str(msg.content):
            agent_result_text = str(msg.content)
            break

    # Create a context message that includes both query and agent result
    if agent_result_text:
        context_message = f"Original Query: {query}\n\n{agent_result_text}"
    else:
        context_message = f"Original Query: {query}"

    messages = [
        SystemMessage(content=prompt_txt),
        HumanMessage(content=context_message),
    ]

    response = llm.invoke(messages)

    return {"chat_history": [response], "response": response}


builder = StateGraph(State)

builder.add_node("call_model", call_model)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

summary_agent_graph = builder.compile(name="SummaryAgent")


def invoke_summary_agent(
    query: str, chat_history: list = [], agent_result: dict = None
):
    """Invoke the Summary Agent to summarize results."""
    if agent_result:
        result_data = agent_result
        if isinstance(agent_result, dict) and "result" in agent_result:
            try:
                import ast

                result_str = agent_result["result"]
                if isinstance(result_str, str) and result_str.startswith("{"):
                    result_data = ast.literal_eval(result_str)
                else:
                    result_data = result_str
            except:  # noqa: E722
                result_data = agent_result

        context_message = f"Original Query: {query}\n\nAgent Result: {result_data}"
        chat_history.append(HumanMessage(content=context_message))

    invoke_data = {
        "query": query,
        "chat_history": chat_history,
    }
    return summary_agent_graph.invoke(invoke_data)


if __name__ == "__main__":
    chat_history = [
        HumanMessage(content="I need a summary of my product search results"),
        AIMessage(
            content="I'll help you summarize your product search results. What did you find?"
        ),
    ]

    agent_result = {
        "agent": "ProductFilterAgent",
        "products": [
            {"name": "Rose Gold Ring", "price": 45000, "category": "Rings"},
            {"name": "Diamond Earrings", "price": 38000, "category": "Earrings"},
        ],
    }

    result = invoke_summary_agent(
        "Show me rings in rose gold under 50k", chat_history, agent_result
    )
    print(f"Summary Result: {result}")

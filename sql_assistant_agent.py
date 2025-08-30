import os
import mysql.connector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from state import State
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CaratLane Stage Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


def get_db_connection():
    """Establish and return a connection to the CaratLane stage MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        return conn
    except Error as e:
        print(f"Error connecting to CaratLane Stage Database: {e}")
        return None


@tool
def execute_sql_query(query: str):
    """Execute a SQL query on the product table (SELECT, DESCRIBE, SHOW only for safety)."""
    query_upper = query.strip().upper()

    # Security check - allow SELECT, DESCRIBE, and SHOW queries
    allowed_prefixes = ["SELECT", "DESCRIBE", "SHOW"]
    if not any(query_upper.startswith(prefix) for prefix in allowed_prefixes):
        return "Error: Only SELECT, DESCRIBE, and SHOW queries are allowed for security reasons."

    # Additional security checks for dangerous operations
    forbidden_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
    ]
    if any(keyword in query_upper for keyword in forbidden_keywords):
        return "Error: Query contains forbidden keywords. Only SELECT, DESCRIBE, and SHOW queries are allowed."

    conn = get_db_connection()
    if not conn:
        return "Unable to connect to CaratLane stage database"

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cursor.description]

        cursor.close()
        conn.close()

        # Format results
        if results:
            formatted_results = []
            for row in results:
                formatted_results.append(dict(zip(column_names, row)))
            return {
                "count": len(formatted_results),
                "results": formatted_results,
                "query": query,
            }
        else:
            return {
                "count": 0,
                "results": [],
                "query": query,
                "message": "Query executed successfully but returned no results.",
            }

    except Error as e:
        print(f"Error executing query: {e}")
        if conn:
            conn.close()
        return f"Error executing query: {e}"


# Create tools list
tools = [execute_sql_query]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

def call_model(state: State):
    """Call the LLM with the current state and tools."""
    user_query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    # Check if we already have schema information
    has_schema = False
    for msg in chat_history:
        if hasattr(msg, "type") and msg.type == "tool" and "Field" in str(msg.content):
            has_schema = True
            break

    if has_schema:
        # We already have schema, generate the final query
        system_prompt = f"""You are a SQL Assistant. You already have the database schema.

Now generate the FINAL SQL query for the user's question: "{user_query}"

Use the correct column names from the schema and these exact values:

**jewellery_type**: Rings, Earrings, Pendants, Necklaces, Bangles, Mangalsutra, Tanmaniya, Silver Rakhi, Bracelets, Nose pin, Mount-Rings, Mount-Earrings, Mount-Pendants, Kada, Charms, Chains, Toe Rings, Anklets, Set Product, Nose Accessories, Silver Articles, Adjustable Rings, Hair Accessories, CuffLinks, Brooch, Brooches, Sets, Watch Charms, Nacklace, Silver Coin, Gold Coin, Baby Bangles, Wrist Watches, Arm Bands, Waist Bands, Earring, Dummy Product, Kanoti, Hasli Necklaces, Kurta Buttons, Bracelet, Charm Builders, Nath

**metal**: 14 KT White, 18 KT Yellow, 18 KT White, 14 KT Two Tone, 14 KT Yellow, 18 KT Two Tone, 14 KT Rose, Silver 925 Silver, 18 KT Rose, Platinum 950 Platinum, 22 KT Yellow, Platinum 950 White, 14 KT, 18 KT Three Tone, Brass Silver, Platinum 950 18 KT Two Tone, 14 KT Three Tone, 18 KT, 9 KT Yellow, 14 KT S925 Yellow, 10 KT Yellow, Silver 925 Yellow, Silver 999 Silver, Silver 925 White, Platinum 950 Two Tone, 22 KT Two Tone, 24 KT Yellow, Silver 925 Rose, 10 KT Rose, Platinum 950 18 KT Three Tone, 22 KT White, 9 KT Rose, Platinum 950 18 KT Two Tone Platinum Rose, 14 KT S925 White, 18 KT S925 White, 18 KT S925 Rose, 18 KT S925 Yellow, 10 KT White, 9 KT White

**purity**: 14 KT, 18 KT, Silver 925, Platinum 950, 22 KT, Brass, Platinum 950 18 KT, 9 KT, 14 KT S925, 10 KT, Silver 999, 24 KT, 18 KT S925

**relationship**: Grandparent, Wife, Girlfriend, Husband, Sister, Others, Daughter, Son, Father, Niece/Nephew, Mother, Friend, Self

**occasion**: anniversary, diwali, christmas, dhanteras, valentines_day, mothers_day, general_gifting, akshaya_tritiya, wedding_season, raksha_bandhan, karva_chauth, ganesh_chaturthi, fathers_day, navratri, new_year

**USER QUERY MAPPING**:
- "ring for my grandpa" → jewellery_type = "Rings" AND relationship = "Grandparent"
- "earrings for wife" → jewellery_type = "Earrings" AND relationship = "Wife"
- "14 KT white gold" → metal = "14 KT White"
- "anniversary gift" → occasion = "anniversary"
- "under 50k" → price < 50000

CRITICAL: Do NOT return SQL text. You MUST call execute_sql_query with your SQL query.

Example: Call execute_sql_query with "SELECT COUNT(*) FROM product WHERE jewellery_type = 'Rings' AND relationship = 'Grandparent' AND price < 20000"

DO NOT use markdown formatting. Call the tool directly."""
    else:
        # First time, get the schema
        system_prompt = f"""You are a SQL Assistant for CaratLane's product database.

FIRST STEP: Call execute_sql_query with "DESCRIBE product" to get the current schema.

User's question: "{user_query}"

After getting the schema, you will be called again to generate the final query."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    response = llm_with_tools.invoke(messages)

    return {"chat_history": [response], "response": response}


def should_continue(state: State):
    """Determine if the agent should continue or end."""
    messages = state["chat_history"]
    last_message = messages[-1]

    # If there are tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tool"

    # If this is a tool message (result from previous tool), check if we need to continue
    if hasattr(last_message, "type") and last_message.type == "tool":
        content = str(last_message.content)

        # Check if this tool result contains actual data (not just schema)
        if "Field" in content and "Type" in content:
            return "call_model"
        elif "count" in content and ("results" in content or "message" in content):
            return END
        else:
            return END

    # If this is an AI message without tool calls, we're done
    return END


def execute_tool(state: State):
    """Execute the tool that was called by the LLM."""
    messages = state["chat_history"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Find and execute the tool
        for tool in tools:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(tool_args)
                    tool_message = ToolMessage(
                        content=str(result), tool_call_id=tool_call["id"]
                    )
                    return {"chat_history": [tool_message]}
                except Exception as e:
                    error_message = ToolMessage(
                        content=f"Error executing {tool_name}: {str(e)}",
                        tool_call_id=tool_call["id"],
                    )
                    return {"chat_history": [error_message]}

    return {"chat_history": []}


# Create the SQL Assistant graph
builder = StateGraph(State)

# Add nodes
builder.add_node("call_model", call_model)
builder.add_node("execute_tool", execute_tool)

# Add edges
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["execute_tool", END])
builder.add_conditional_edges("execute_tool", should_continue, ["call_model", END])

# Compile the graph
sql_assistant_graph = builder.compile(name="SQLAssistantAgent")


def invoke_sql_assistant(query: str, chat_history: list = []):
    """Invoke the SQL Assistant Agent."""
    invoke_data = {
        "query": query,
        "chat_history": chat_history,
    }
    result = sql_assistant_graph.invoke(invoke_data)

    # Extract the actual result from the tool execution
    if result.get("chat_history"):
        last_message = result["chat_history"][-1]
        if hasattr(last_message, "content"):
            # If it's a tool message with results, return that
            if "count" in last_message.content or "results" in last_message.content:
                try:
                    # Try to parse the content as JSON if it's a tool result
                    import json

                    parsed_content = json.loads(last_message.content)
                    return parsed_content
                except:  # noqa: E722
                    # If not JSON, return the content as is
                    return {"result": last_message.content}

    # Fallback: return the full result
    return result


if __name__ == "__main__":
    # Test the SQL Assistant
    chat_history = [
        HumanMessage(content="Hi, I need to query the product table"),
        AIMessage(
            content="Hello! I can help you write and execute SQL queries on the product table. What would you like to know?"
        ),
    ]

    test_queries = [
        "Show me all rings",
        "How many products do we have?",
        "What's the average price?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print(f"{'=' * 50}")
        result = invoke_sql_assistant(query, chat_history)
        print(f"Result: {result}")

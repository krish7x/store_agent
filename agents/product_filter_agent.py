import logging
from llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from state.state import State
from helpers.db_helper import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = get_llm()


@tool
def execute_sql_query(query: str) -> str:
    """Execute a SQL query on the product table (SELECT, DESCRIBE, SHOW only for safety)."""
    query_upper = query.strip().upper()

    # Security check - allow SELECT, DESCRIBE, and SHOW queries
    allowed_prefixes = ["SELECT", "DESCRIBE", "SHOW"]
    if not any(query_upper.startswith(prefix) for prefix in allowed_prefixes):
        return str(
            {
                "count": 0,
                "results": [],
                "query": query,
                "message": "Error: Only SELECT, DESCRIBE, and SHOW queries are allowed for security reasons.",
            }
        )

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
        return str(
            {
                "count": 0,
                "results": [],
                "query": query,
                "message": "Error: Query contains forbidden keywords. Only SELECT, DESCRIBE, and SHOW queries are allowed.",
            }
        )

    try:
        # Connect to the database using the existing function
        conn = get_db_connection()
        if not conn:
            import json

            return json.dumps(
                {
                    "count": 0,
                    "results": [],
                    "query": query,
                    "message": "Unable to connect to database",
                }
            )

        cursor = conn.cursor(dictionary=True)

        # Execute the query
        cursor.execute(query)

        # Fetch results
        results = cursor.fetchall()

        # Close connection
        cursor.close()
        conn.close()

        # Format results
        if results:
            # Count total results
            count = len(results)

            # Determine result limit based on user request
            # Default to 10 results for readability, but check if user asked for specific number
            result_limit = 10

            # First, check if the SQL query itself has a LIMIT clause
            if "LIMIT" in query_upper:
                try:
                    # Find LIMIT clause and extract number
                    limit_match = query_upper.split("LIMIT")[-1].strip().split()[0]
                    if limit_match.isdigit():
                        # If SQL has LIMIT, respect it and show all results up to that limit
                        result_limit = int(limit_match)
                except:  # noqa: E722
                    pass
            else:
                # Check if user asked for specific number of results in natural language
                user_query = query.lower()
                if "top" in user_query or "show" in user_query or "limit" in user_query:
                    # User might have asked for "top 5", "show 3", "limit 15", etc.
                    import re

                    # Look for patterns like "top 5", "show 10", "limit 20"
                    patterns = [
                        r"top\s+(\d+)",
                        r"show\s+(\d+)",
                        r"limit\s+(\d+)",
                        r"(\d+)\s+results?",
                        r"(\d+)\s+items?",
                        r"(\d+)\s+products?",
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, user_query)
                        if match:
                            requested_count = int(match.group(1))
                            result_limit = min(requested_count, count)
                            break

            # Format the response
            response = {
                "count": count,  # Total available results
                "results": results[:result_limit],  # Show limited results
                "query": query,
                "message": f"Query executed successfully. Found {count} total results. Showing {min(result_limit, count)} results.",
                "total_available": count,  # Total count available
                "showing": min(result_limit, count),  # How many we're showing
                "has_more": count
                > result_limit,  # Flag indicating if there are more results
            }

            import json

            return json.dumps(response)
        else:
            import json

            return json.dumps(
                {
                    "count": 0,
                    "results": [],
                    "query": query,
                    "message": "Query executed successfully. No results found.",
                }
            )

    except Exception as e:
        logger.error(f"Database error: {e}")
        import json

        return json.dumps(
            {
                "count": 0,
                "results": [],
                "query": query,
                "message": f"Error executing query: {str(e)}",
            }
        )


# List of available tools
tools = [execute_sql_query]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


def product_filter_node(state: State):
    """Call the LLM to generate SQL queries based on user input."""
    try:
        chat_history = state.get("chat_history", [])
        query = state.get("query", "")

        # Create system prompt for SQL generation
        system_prompt = """You are a SQL expert for CaratLane's product database.

        Your task is to generate SQL queries based on user requests for product information.

        Available tables:
        - product: Contains product details like sku, price, metal, purity, jewellery_type, etc.

        **EXACT COLUMN VALUES - USE THESE EXACTLY:**

        **jewellery_type**: Rings, Earrings, Pendants, Necklaces, Bangles, Mangalsutra, Tanmaniya, Silver Rakhi, Bracelets, Nose pin, Mount-Rings, Mount-Earrings, Mount-Pendants, Kada, Charms, Chains, Toe Rings, Anklets, Set Product, Nose Accessories, Silver Articles, Adjustable Rings, Hair Accessories, CuffLinks, Brooch, Brooches, Sets, Watch Charms, Nacklace, Silver Coin, Gold Coin, Baby Bangles, Wrist Watches, Arm Bands, Waist Bands, Earring, Dummy Product, Kanoti, Hasli Necklaces, Kurta Buttons, Bracelet, Charm Builders, Nath

        **metal**: 14 KT White, 18 KT Yellow, 18 KT White, 14 KT Two Tone, 14 KT Yellow, 18 KT Two Tone, 14 KT Rose, Silver 925 Silver, 18 KT Rose, Platinum 950 Platinum, 22 KT Yellow, Platinum 950 White, 14 KT, 18 KT Three Tone, Brass Silver, Platinum 950 18 KT Two Tone, 14 KT Three Tone, 18 KT, 9 KT Yellow, 14 KT S925 Yellow, 10 KT Yellow, Silver 925 Yellow, Silver 999 Silver, Silver 925 White, Platinum 950 Two Tone, 22 KT Two Tone, 24 KT Yellow, Silver 925 Rose, 10 KT Rose, Platinum 950 18 KT Three Tone, 22 KT White, 9 KT Rose, Platinum 950 18 KT Two Tone Platinum Rose, 14 KT S925 White, 18 KT S925 White, 18 KT S925 Rose, 18 KT S925 Yellow, 10 KT White, 9 KT White

        **purity**: 14 KT, 18 KT, Silver 925, Platinum 950, 22 KT, Brass, Platinum 950 18 KT, 9 KT, 14 KT S925, 10 KT, Silver 999, 24 KT, 18 KT S925

        **relationship**: Grandparent, Wife, Girlfriend, Husband, Sister, Others, Daughter, Son, Father, Niece/Nephew, Mother, Friend, Self

        **occasion**: anniversary, diwali, christmas, dhanteras, valentines_day, mothers_day, general_gifting, akshaya_tritiya, wedding_season, raksha_bandhan, karva_chauth, ganesh_chaturthi, fathers_day, navratri, new_year

        **USER QUERY MAPPING EXAMPLES**:
        - "ring for my grandpa" → jewellery_type = "Rings" AND relationship = "Grandparent"
        - "earrings for wife" → jewellery_type = "Earrings" AND relationship = "Wife"
        - "14 KT white gold" → metal = "14 KT White"
        - "anniversary gift" → occasion = "anniversary"
        - "under 50k" → price < 50000
        - "silver vs gold options" → metal IN ('Silver 925', '14 KT White', '18 KT Yellow', '18 KT White', '14 KT Rose', '18 KT Rose')

        Guidelines:
        - Use 'jewellery_type' NOT 'category' for product types
        - Use 'sku' instead of 'name' or 'id' for product identification
        - Generate valid SQL queries using the EXACT column values above
        - Use appropriate WHERE clauses for filtering
        - Return results in a readable format
        - Focus on product search and filtering queries

        Result Handling:
        - By default, show 10 results for better readability
        - If user asks for specific number (e.g., "top 5", "show 3"), add LIMIT clause
        - For comparison queries (e.g., "silver vs gold", "compare options"), use LIMIT 20-50 to show variety
        - For broad category searches, consider LIMIT 15-25 for better representation
        - Always return total count along with limited results
        - Use ORDER BY price ASC/DESC when appropriate for price-based queries
        - For expensive items, consider ORDER BY price DESC
        - For budget items, consider ORDER BY price ASC

        Examples:
        - "Show me top 5 expensive rings" → SELECT ... ORDER BY price DESC LIMIT 5
        - "Show me 3 cheapest earrings" → SELECT ... ORDER BY price ASC LIMIT 3
        - "Show me rings under 50k" → SELECT ... LIMIT 10 (default)
        - "Compare silver vs gold options" → SELECT ... LIMIT 25 (for variety)
        - "Show me budget options" → SELECT ... LIMIT 20 (for comparison)

        Always use the execute_sql_query tool to run your SQL queries."""

        # Create messages for the LLM
        messages = [SystemMessage(content=system_prompt)]

        # Add the chat history to provide context
        messages.extend(chat_history)

        # Add the user query
        if query:
            messages.append(HumanMessage(content=query))

        # Generate SQL query using LLM with tools
        response = llm_with_tools.invoke(messages)

        # Append the response to chat history instead of replacing it
        updated_chat_history = chat_history + [response]

        logger.info("SQL query generated successfully")

        return {"chat_history": updated_chat_history, "response": response}

    except Exception as e:
        logger.error(f"Error in SQL generation: {e}")
        # Return the original chat history if there's an error
        return {"chat_history": chat_history, "response": None}


def query_executor_node(state: State):
    """Execute the tool calls from the LLM response."""
    try:
        chat_history = state.get("chat_history", [])

        # Get the last message which should have tool calls
        if not chat_history:
            logger.warning("No chat history to execute tools from")
            return {"chat_history": chat_history}

        last_message = chat_history[-1]

        # Check if the message has tool calls
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.info("No tool calls to execute")
            return {"chat_history": chat_history}

        # Execute each tool call
        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            # Execute the tool
            if tool_name == "execute_sql_query":
                query = tool_args.get("query", "")
                result = execute_sql_query(query)
                tool_results.append(result)

                # Create a tool message
                tool_message = ToolMessage(content=result, tool_call_id=tool_call["id"])

                # Append the tool message to chat history
                updated_chat_history = chat_history + [tool_message]

                logger.info(f"Tool execution completed: {tool_name}")

                return {"chat_history": updated_chat_history, "response": tool_message}

        # If no tools were executed, return original chat history
        return {"chat_history": chat_history}

    except Exception as e:
        logger.error(f"Error executing tools: {e}")
        return {"chat_history": chat_history}

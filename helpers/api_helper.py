import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_summary_content(orchestrator_result: dict) -> str:
    """Extract summary content from orchestrator result."""
    if not orchestrator_result or not isinstance(orchestrator_result, dict):
        return "No summary available"

    # Get the final response from the summary generation
    response = orchestrator_result.get("response")
    if response and hasattr(response, "content"):
        return str(response.content)

    # Fallback: look for summary in chat history
    chat_history = orchestrator_result.get("chat_history", [])
    for msg in reversed(chat_history):  # Start from the end
        if hasattr(msg, "content") and msg.content:
            content = str(msg.content)
            if "rings" in content.lower() or "products" in content.lower():
                return content

    return "Summary not available"


def extract_result_content(orchestrator_result: dict) -> tuple:
    """Extract result content and SQL query from orchestrator result."""
    if not orchestrator_result or not isinstance(orchestrator_result, dict):
        return {}, "Query executed"

    sql_query = "Query executed"
    result_data = {}

    # Look for tool results in chat history
    chat_history = orchestrator_result.get("chat_history", [])

    # Find the AIMessage that contains the SQL execution results (now using AIMessage instead of ToolMessage)
    # Iterate in reverse to get the most recent results
    for msg in reversed(chat_history):
        if hasattr(msg, "type") and msg.type == "ai":
            content = str(msg.content)
            # This should contain the SQL results with count, results, query
            if "SQL Query Result:" in content:
                try:
                    import json

                    # Extract the JSON part after "SQL Query Result: "
                    json_start = content.find("SQL Query Result: ") + len("SQL Query Result: ")
                    json_content = content[json_start:].strip()

                    # Parse the JSON content from the tool message
                    parsed_result = json.loads(json_content)
                    if isinstance(parsed_result, dict):
                        # Extract the SQL query
                        if "query" in parsed_result:
                            sql_query = parsed_result["query"]

                        # Extract the actual data
                        result_data = {
                            "count": parsed_result.get("count", 0),
                            "results": parsed_result.get("results", []),
                            "message": parsed_result.get("message", ""),
                            "total_available": parsed_result.get(
                                "total_available", parsed_result.get("count", 0)
                            ),
                            "showing": parsed_result.get(
                                "showing", parsed_result.get("count", 0)
                            ),
                            "has_more": parsed_result.get("has_more", False),
                        }
                        return result_data, sql_query

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool result as JSON: {e}")
                    logger.warning(f"Content was: {content}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to parse tool result: {e}")
                    logger.warning(f"Content was: {content}")
                    continue

    # If we didn't find tool results, return fallback
    return {"message": "User query processed successfully"}, sql_query

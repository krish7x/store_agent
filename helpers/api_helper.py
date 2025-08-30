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

    # Find the ToolMessage that contains the SQL execution results
    for msg in chat_history:
        if hasattr(msg, "type") and msg.type == "tool":
            content = str(msg.content)
            # This should contain the SQL results with count, results, query
            if "count" in content and "results" in content:
                try:
                    import json

                    # Parse the JSON content from the tool message
                    parsed_result = json.loads(content)
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
                    continue
                except Exception as e:
                    logger.warning(f"Failed to parse tool result: {e}")
                    continue

    # If we didn't find tool results, return fallback
    return {"message": "Query processed successfully"}, sql_query

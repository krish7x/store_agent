import json
import redis
import os
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)


class SimpleRedisChatHistory:
    """Simple Redis chat history implementation that works with regular Redis."""

    def __init__(self, redis_url: str, session_id: str):
        self.redis_client = redis.from_url(redis_url)
        self.session_id = session_id
        self.key = f"chat:{session_id}"

    def add_message(self, message: BaseMessage):
        """Add a single message to the chat history."""
        self.add_messages([message])

    def add_messages(self, messages: List[BaseMessage]):
        """Add multiple messages to the chat history."""
        try:
            # Get existing messages
            existing_messages = self.get_messages()

            # Add new messages
            all_messages = existing_messages + messages

            # Serialize all messages to JSON array
            messages_json = json.dumps(
                [self._serialize_message(msg) for msg in all_messages]
            )

            # Store as single JSON string in Redis
            self.redis_client.set(self.key, messages_json)
            logger.info(
                f"Added {len(messages)} messages to Redis (total: {len(all_messages)})"
            )
        except Exception as e:
            logger.error(f"Failed to add messages to Redis: {e}")
            raise

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve all messages from the chat history."""
        try:
            # Get the JSON string from Redis
            messages_json = self.redis_client.get(self.key)

            if not messages_json:
                return []

            # Parse the JSON array
            message_dicts = json.loads(messages_json)
            messages = []

            for message_dict in message_dicts:
                try:
                    message = self._deserialize_message(message_dict)
                    messages.append(message)
                except Exception as e:
                    logger.warning(f"Failed to deserialize message: {e}")
                    continue

            logger.info(f"Retrieved {len(messages)} messages from Redis")
            return messages
        except Exception as e:
            logger.error(f"Failed to get messages from Redis: {e}")
            return []

    def clear(self):
        """Clear all messages from the chat history."""
        try:
            self.redis_client.delete(self.key)
            logger.info("Cleared chat history from Redis")
        except Exception as e:
            logger.error(f"Failed to clear chat history: {e}")

    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        """Serialize a LangChain message to a compact dictionary."""
        # Base message structure
        message_dict = {
            "type": message.type,
            "content": message.content,
        }

        # For AI messages with SQL results, extract and store only the SQL query
        if (
            message.type == "ai"
            and hasattr(message, "content")
            and message.content
            and "SQL Query Result:" in str(message.content)
        ):
            try:
                import json

                content_str = str(message.content)
                json_start = content_str.find("SQL Query Result: ") + len(
                    "SQL Query Result: "
                )
                json_content = content_str[json_start:].strip()

                # Try to parse JSON, if it fails, use regex extraction
                try:
                    result_data = json.loads(json_content)
                    sql_query = result_data.get("query", "")
                except json.JSONDecodeError:
                    # Use regex to extract query from malformed JSON
                    import re

                    # More robust regex to handle escaped quotes and complex SQL
                    query_match = re.search(
                        r'"query":\s*"([^"]*(?:\\.[^"]*)*)"', json_content, re.DOTALL
                    )
                    if query_match:
                        sql_query = (
                            query_match.group(1)
                            .replace('\\"', '"')
                            .replace("\\n", " ")
                            .replace("\\t", " ")
                        )
                    else:
                        # Fallback: try to find query between quotes more broadly
                        query_match = re.search(r'"query":\s*"([^"]+)"', json_content)
                        if query_match:
                            sql_query = query_match.group(1)
                        else:
                            sql_query = ""

                if sql_query:
                    message_dict["sql_query"] = sql_query
                    message_dict["content"] = f"SQL Query executed: {sql_query}"
            except Exception as e:
                logger.warning(f"Could not extract SQL query from message: {e}")

        # For tool calls, store only essential information
        if hasattr(message, "tool_calls") and message.tool_calls:
            # Extract only the query from tool calls
            tool_calls = []
            for tool_call in message.tool_calls:
                if (
                    isinstance(tool_call, dict)
                    and tool_call.get("name") == "execute_sql_query"
                ):
                    args = tool_call.get("args", {})
                    if "query" in args:
                        tool_calls.append(
                            {"name": "execute_sql_query", "query": args["query"]}
                        )
                else:
                    tool_calls.append(tool_call)
            message_dict["tool_calls"] = tool_calls

        return message_dict

    def _deserialize_message(self, message_dict: Dict[str, Any]) -> BaseMessage:
        """Deserialize a dictionary back to a LangChain message."""
        msg_type = message_dict.get("type", "human")
        content = message_dict.get("content", "")

        # Create the appropriate message type
        if msg_type == "human":
            return HumanMessage(content=content)
        elif msg_type == "ai":
            # For AI messages, reconstruct the full content if we have SQL query
            if "sql_query" in message_dict:
                # Reconstruct the SQL Query Result format for pagination analysis
                sql_query = message_dict["sql_query"]
                reconstructed_content = f'SQL Query Result: {{"query": "{sql_query}", "count": 0, "results": []}}'
                return AIMessage(content=reconstructed_content)
            else:
                return AIMessage(content=content)
        elif msg_type == "system":
            return SystemMessage(content=content)
        else:
            # Fallback to AIMessage for unknown types
            return AIMessage(content=content)


def get_simple_chat_history(session_id: str) -> SimpleRedisChatHistory:
    """Get a simple Redis chat history instance."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    return SimpleRedisChatHistory(redis_url, session_id)

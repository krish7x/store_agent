from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn
import logging

# Import our agents
from orchestrator_agent import invoke_orchestrator
from summary_agent import invoke_summary_agent
from sql_assistant_agent import invoke_sql_assistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CartaLane Inventory Operations Chatbot API",
    description="AI-powered inventory management system with intelligent agent routing",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """Request model for chat interactions."""

    query: str
    chat_history: Optional[List[Dict[str, str]]] = []
    cart: Optional[List[Dict[str, Any]]] = []
    store_code: Optional[str] = ""


class ChatResponse(BaseModel):
    """Response model for chat interactions."""

    summary: str
    query: str
    result: Any


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str
    timestamp: str


def convert_chat_history(chat_history: List[Dict[str, str]]) -> List:
    """Convert chat history from dict format to LangChain message format."""
    converted_history = []

    for message in chat_history:
        if message.get("role") == "user":
            converted_history.append(HumanMessage(content=message.get("content", "")))
        elif message.get("role") == "assistant":
            converted_history.append(AIMessage(content=message.get("content", "")))

    return converted_history


def invoke_specific_agent(agent_name: str, query: str, chat_history: List, **kwargs):
    """Invoke the specific agent based on the agent name."""
    try:
        if agent_name == "ProductFilterAgent":
            return invoke_sql_assistant(query, chat_history)
        else:
            raise ValueError(f"Unknown agent: {agent_name}")
    except Exception as e:
        logger.error(f"Error invoking {agent_name}: {str(e)}")
        raise


def extract_summary_content(summary_result: dict) -> str:
    """Extract summary content from the summary agent response."""
    if not summary_result or not isinstance(summary_result, dict):
        return ""

    # Check for response key first
    if "response" in summary_result:
        response_obj = summary_result["response"]
        if hasattr(response_obj, "content"):
            return response_obj.content
        return str(response_obj)

    # Check for chat_history key
    elif "chat_history" in summary_result:
        chat_history = summary_result["chat_history"]
        if chat_history and len(chat_history) > 0:
            last_message = chat_history[-1]
            if hasattr(last_message, "content"):
                return last_message.content
            return str(last_message)

    return ""


def extract_result_content(agent_result: dict) -> tuple:
    """Extract result content and query from agent result."""
    if not agent_result or not isinstance(agent_result, dict):
        return {}, "Query executed"

    if "result" in agent_result:
        result_str = agent_result["result"]

        try:
            import ast

            parsed_result = ast.literal_eval(result_str)

            if isinstance(parsed_result, dict):
                # Remove the query from the nested result to avoid duplication
                cleaned_result = parsed_result.copy()
                if "query" in cleaned_result:
                    del cleaned_result["query"]
                return cleaned_result, parsed_result.get("query", "Unknown query")
            return parsed_result, "Query executed"
        except Exception:
            return result_str, "Query executed"

    return agent_result, "Query executed"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from datetime import datetime

    return HealthResponse(
        status="healthy",
        message="CartaLane Inventory Operations Chatbot API is running",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that implements the complete workflow:
    1. User query → 2. Orchestrator → 3. Specific Agent → 4. Summary → 5. Response
    """
    try:
        logger.info(f"Received chat request: {request.query}")

        # Step 1: Convert chat history to LangChain format
        langchain_chat_history = convert_chat_history(request.chat_history)

        # Step 2: Route query through orchestrator
        logger.info("Routing query through orchestrator...")
        orchestrator_result = invoke_orchestrator(
            query=request.query,
            chat_history=langchain_chat_history,
            cart=request.cart,
            store_code=request.store_code,
        )

        selected_agent = orchestrator_result.get("selected_agent", "ProductFilterAgent")
        logger.info(f"Orchestrator selected agent: {selected_agent}")

        # Step 3: Invoke the specific agent
        logger.info(f"Invoking {selected_agent}...")
        agent_result = invoke_specific_agent(
            agent_name=selected_agent,
            query=request.query,
            chat_history=langchain_chat_history,
            cart=request.cart,
            store_code=request.store_code,
        )

        logger.info(f"Agent result received from {selected_agent}")

        # Step 4: Generate summary using summary agent
        logger.info("Generating summary...")
        summary_result = invoke_summary_agent(
            query=request.query,
            chat_history=langchain_chat_history,
            agent_result={"agent": selected_agent, "result": agent_result},
        )

        # Extract summary content
        summary_content = extract_summary_content(summary_result)

        # Extract result content and query
        result_content, query_content = extract_result_content(agent_result)

        # Step 5: Prepare response
        response = ChatResponse(
            summary=summary_content, result=result_content, query=query_content
        )

        logger.info("Chat request processed successfully")
        return response

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "api_controller:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )

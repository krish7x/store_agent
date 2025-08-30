import logging
from fastapi import FastAPI, HTTPException
from agents.orchestrator_agent import invoke_orchestrator
from helpers.api_helper import extract_summary_content, extract_result_content
from api_model import ChatResponse, ChatRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CaratLane Store Agent API")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request through the orchestrator."""
    try:
        logger.info(f"Processing query: {request.query}")

        # Invoke the orchestrator
        orchestrator_result = invoke_orchestrator(request.query)

        if "error" in orchestrator_result:
            raise HTTPException(status_code=500, detail=orchestrator_result["error"])

        # Extract summary and results
        summary = extract_summary_content(orchestrator_result)
        result_data, sql_query = extract_result_content(orchestrator_result)

        logger.info(f"Query: {sql_query}")
        logger.info(f"Results: {result_data}")
        logger.info(f"Summary: {summary}")

        return ChatResponse(summary=summary, query=sql_query, result=result_data)

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

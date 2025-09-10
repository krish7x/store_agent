from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    user_email: str


class ChatResponse(BaseModel):
    summary: str
    query: str
    result: dict

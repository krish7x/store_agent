from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    """Get the LLM model."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    return llm

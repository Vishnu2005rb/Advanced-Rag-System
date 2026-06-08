import os
from typing import Optional
from langchain_groq import ChatGroq

def get_groq_llm(
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    api_key: Optional[str] = None
) -> ChatGroq:
    """
    Initializes and returns a ChatGroq LLM instance.
    Falls back to the GROQ_API_KEY env variable if no custom api_key is supplied.
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")
        
    if not api_key:
        raise ValueError(
            "Groq API Key is missing. Please set GROQ_API_KEY in the environment or pass it in the request."
        )
        
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=api_key
    )

from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from rag.llm import get_groq_llm

def rewrite_query(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = "llama-3.3-70b-versatile",
    api_key: Optional[str] = None
) -> str:
    """
    Rewrites the user question into an optimized keyword-rich search query
    based on conversation history.
    """
    if not chat_history:
        chat_history = []
        
    # Format chat history for prompt context
    history_str = ""
    for msg in chat_history[-4:]: # Use last 4 messages to save context space
        role = "User" if msg.get("role") == "user" else "AI"
        history_str += f"{role}: {msg.get('text')}\n"
        
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert search query optimizer. 
        Given the following conversation history and a user's latest question, rewrite the question into a single search query that is optimized for a vector database similarity search.
        
        Focus on:
        - Resolving coreferences (e.g. replacing 'it', 'they', 'this' with the actual nouns they refer to).
        - Adding relevant keywords from the context.
        - Removing conversational filler (e.g. 'can you tell me', 'please explain').
        - Output ONLY the optimized query string. Do NOT add any preamble, introduction, or explanations.
        
        Conversation History:
        {history}
        
        User's Question:
        {question}
        
        Optimized Search Query:
        """
    )
    
    try:
        llm = get_groq_llm(model_name=model_name, temperature=0.0, api_key=api_key)
        chain = prompt | llm
        response = chain.invoke({
            "history": history_str or "No history yet.",
            "question": question
        })
        rewritten = response.content.strip()
        # Clean up any surrounding quotes LLM might add
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        return rewritten
    except Exception as e:
        print(f"Query rewrite failed: {e}. Falling back to original question.")
        return question

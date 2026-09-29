from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
from rag_core import openai_completions_deployment
from ai_agent import get_response_from_ai_agent
import logging

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# Deployments available in this Azure OpenAI resource.
# The default is the CFTC completions deployment from .env.
# Includes the original Groq model names for backward compatibility.
ALLOWED_MODEL_NAMES = [
    openai_completions_deployment or "gpt-5-csl14",
    "gpt-4o-mini",
    "gpt-4o",
    "llama-3.1-8b-instant",
    "mistral-8x7b-32786",
    "llama-3.3-70b-versatile",
]

app = FastAPI(title="AI Agent")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/health")
def health_endpoint():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/models")
def models_endpoint():
    """Return the list of allowed model deployments."""
    return {"allowed_models": ALLOWED_MODEL_NAMES}


@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using the Azure OpenAI agent
    and the Cosmos DB retrieval tool.
    """
    try:
        llm_id = request.model_name
        query = request.messages
        allow_search = request.allow_search
        system_prompt = request.system_prompt
        provider = request.model_provider

        logger.info(f"Received request: {request}")

        if request.model_name not in ALLOWED_MODEL_NAMES:
            raise HTTPException(status_code=400, detail="Model name not allowed. Please select a valid model")

        # Create AI Agent and fetch a response from it
        response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
        logger.info(f"AI Agent response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

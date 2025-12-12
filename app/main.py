import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .db import ping_database
from .agent import answer_question, clear_chat_history


class ChatRequest(BaseModel):
    question: str
    session_id: str = None


class NewChatRequest(BaseModel):
    session_id: str


app = FastAPI(title="AI Procurement Assistant")

# Get the project root directory (parent of app directory)
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the chat interface."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/health")
async def health_check():
    db_ok = await ping_database()
    return {"status": "ok" if db_ok else "degraded", "mongo_connected": db_ok}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chat endpoint for the procurement assistant.
    Supports session-based chat history for context tracking.
    """
    response = await answer_question(req.question, session_id=req.session_id)
    return response


@app.post("/new-chat")
async def new_chat(req: NewChatRequest):
    """
    Clear chat history for a session.
    """
    clear_chat_history(req.session_id)
    return {"status": "ok", "message": "Chat history cleared"}



# app.py

import logging
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from RAG.rag import RAG_chatbot

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Instantiate your Side-A RAG chatbot
bot = RAG_chatbot()

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class UserQuery(BaseModel):
    user_query: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat_endpoint(payload: UserQuery = Body(...)):
    try:
        # Synchronous call to bot.ask(); no 'await' needed
        answer = bot.ask(payload.user_query)
        return {"answer": answer}
    except Exception as e:
        logging.error("Error in chat_endpoint: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Run with:
# uvicorn app:app --reload

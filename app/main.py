import os
from collections import defaultdict
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

from app.rag import process_docs, query_docs

# Load .env
load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Gemini client
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Simple session memory: { session_id: [ {role, content}, ... ] }
sessions: dict = defaultdict(list)


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {"msg": "Only plain text (.txt) files are supported."}
    process_docs(text)
    return {"msg": f"✅ '{file.filename}' uploaded and indexed successfully!"}


@app.post("/ask")
async def ask(req: ChatRequest):
    history = sessions[req.session_id]

    # Last 10 turns as conversation context
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-10:]
        )

    # RAG: retrieve relevant chunks from uploaded document
    doc_context = "\n".join(query_docs(req.question))

    prompt = f"""You are a helpful AI assistant. Answer the user's question using the document and conversation history below.

Conversation History:
{history_text if history_text else "None yet."}

Document Context:
{doc_context}

User Question: {req.question}

Instructions:
- Prefer information from the Document Context.
- Use Conversation History for follow-up questions.
- If the answer is not in the document, say: "I don't have that information in the uploaded document."
- Be concise and clear.
"""

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text

    # Save to session memory
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})

    return {"answer": answer}
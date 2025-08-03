from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PipeLine import load_vectorstore, get_combined_answer

# Initialize FastAPI
app = FastAPI(title="Legal Advisor Chatbot")

# Load vectorstore once when server starts
vectorstore = load_vectorstore()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Input schema
class Question(BaseModel):
    question: str

# Homepage
@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint for answering questions

@app.post("/ask/")
def ask_question(query: Question):
    response = get_combined_answer(query.question, vectorstore)
    # Extract content if it's an AIMessage or similar object
    if hasattr(response, 'content'):
        return {"answer": response.content}
    return {"answer": str(response)}

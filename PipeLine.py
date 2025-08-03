import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings 

#from langchain.document_loaders import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import pickle

# Load .env keys
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

VECTORSTORE_DIR = "/Users/coding/AI_projects/LegalAdvisor Chatbot/ipc_faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load FAISS vectorstore
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local("/Users/coding/AI_projects/LegalAdvisor Chatbot/ipc_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Gemini + FAISS
def answer_from_vectorstore(query, vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain.invoke({"query": query})["result"]

# Google Search Tool
def answer_from_google(query):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)
    search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
    search_tool = Tool(name="Google Search", func=search.run, description="Search the web for legal info")
    agent = initialize_agent(tools=[search_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    return agent.run(query)

# Final combined logic
def get_combined_answer(query, vectorstore):
    # Step 1: Get answers
    local_answer = answer_from_vectorstore(query, vectorstore)
    web_answer = answer_from_google(query)

    # Step 2: Let Gemini decide what to do with both
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)

    prompt = f"""
You are a legal advisor chatbot. You have received the following question:

--- Question ---
{query}

You have two sources of information:

--- IPC-Based Answer ---
{local_answer}

--- Web-Based Answer ---
{web_answer}

Instructions:
- Provide the most accurate, human-like, and helpful answer.
- If both sources are useful, combine them naturally and eliminate any repetition or noise.
- If one is clearly better, use only that.
- Keep the tone clear and natural for a layperson.
- Be concise but cover the important details.

Final Answer:
"""

    return llm.invoke(prompt)

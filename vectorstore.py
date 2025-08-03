import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import os

# Load Excel
df = pd.read_csv("/content/drive/MyDrive/ML Projects/LegalAdvisorChatBot/Data/ipc_sections.csv")

# One document per row
docs = []
for idx, row in df.iterrows():
    content = f"""Description: {row['Description']}
Offense: {row['Offense']}
Punishment: {row['Punishment']}
Section: {row['Section']}"""
    docs.append(Document(page_content=content, metadata={"section": row["Section"]}))


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()
# Save vector store to a folder
vectorstore.save_local("ipc_faiss_index")


vectorstore.save_local("/content/drive/MyDrive/ML Projects/LegalAdvisorChatBot/Data/ipc_faiss_index")

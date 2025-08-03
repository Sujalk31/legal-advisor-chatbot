# 🧑‍⚖️ Legal Advisor Chatbot using LangChain & Gemini

A smart legal assistant chatbot that combines **local Indian Penal Code (IPC)** data and **real-time web search** to answer user queries accurately and contextually. This project is built using **LangChain**, **FAISS**, **Gemini 1.5 Flash**, and **Google Search API**, offering a Retrieval-Augmented Generation (RAG) solution tailored for legal queries.

---

## 📌 Use Cases

- 🔍 Look up IPC sections, offenses, and punishments
- 🌐 Get real-time updates on legal topics via web search
- 📘 Assist law students and professionals in research
- 💼 Automate client support for legal firms

---

## ⚙️ Tech Stack

| Technology                    | Purpose / Use                                                           |
| ----------------------------- | ----------------------------------------------------------------------- |
| 🧠 **LangChain**              | LLM orchestration (retriever, tools, chains, agents)                    |
| 🔎 **FAISS**                  | Fast vector similarity search over IPC documents                        |
| 📚 **HuggingFace Embeddings** | Embedding model: `all-MiniLM-L6-v2`                                     |
| 🤖 **Gemini 1.5 Flash**       | LLM for response generation and multi-source reasoning                  |
| 🌐 **Google Search API**      | Real-time search for additional legal context                           |
| 🌟 **FastAPI**                | High-performance backend API for chatbot interaction                    |
| 🧪 **Python-dotenv**          | Securely loads environment variables from `.env`                        |



---

## 🧠 How It Works

1. ✅ Loads a FAISS vectorstore of IPC sections
2. 🔄 Queries Gemini using LangChain's `RetrievalQA` on local data
3. 🌍 Uses Google Search to get real-time info
4. 🧠 Combines both answers using Gemini for best final result

---

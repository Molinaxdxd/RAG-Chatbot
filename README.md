# 🧠 Filipino Sports RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot focused on Filipino sports. Ask about athletes, teams, and tournaments — and get answers grounded in real documents, not hallucinated guesses.

![LangChain + Groq + Qdrant](https://img.shields.io/badge/Stack-LangChain%20%2B%20Groq%20%2B%20Qdrant-blue)

---

## 🚀 Features

- 📄 Ingests selected Filipino athlete and team pages
- 🔍 Fast retrieval using **Qdrant** vector store
- 🧠 LLM inference via **Groq API** (Mixtral-8x7B)
- 💡 Clean, context-grounded responses — no training required
- 🛡️ Secrets handled via `.env` (excluded from Git)

---

## 🧱 Stack

- **LangChain** (RAG pipeline)
- **Groq** (LLM inference, blazing-fast)
- **Qdrant** (vector DB for embedding search)
- **HuggingFace Transformers** (embedding model)
- **Python** + **Streamlit** (if UI added)

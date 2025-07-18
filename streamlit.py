import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load .env
load_dotenv()

# Setup embeddings + Qdrant
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Qdrant(
    client=QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    ),
    collection_name="filipinoboxers",
    embeddings=embeddings
)

# Setup LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# Boxer list
KNOWN_ENTITIES = ["Manny Pacquiao", "Nonito Donaire"]

# Extract entity
def extract_target_entity(query: str):
    for name in KNOWN_ENTITIES:
        if re.search(re.escape(name), query, re.IGNORECASE):
            return name
    return None

# Prompt
prompt = PromptTemplate.from_template(
    """Answer the question based only on the context below.

Context:
{context}

Question: {question}"""
)

output_parser = StrOutputParser()

# Streamlit UI
st.set_page_config(page_title="🥊 Filipino Boxing Chatbot", layout="centered")
st.title("🥊 Filipino Boxers RAG Chatbot")

# Input
query = st.chat_input("Ask me about Manny Pacquiao or Nonito Donaire...")

if query:
    target = extract_target_entity(query)

    # Use filtered retriever
    if target:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 8, "filter": {"athlete": target}}
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt | llm | output_parser
    answer = chain.invoke({"context": context, "question": query})

    # Display current query and answer (no history)
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if docs:
            with st.expander("📄 Sources"):
                for doc in docs:
                    st.markdown(f"- {doc.page_content.strip()}")

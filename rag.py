import os
import re
from dotenv import load_dotenv
from langchain_qdrant import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to Qdrant
vectorstore = Qdrant(
    client=QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    ),
    collection_name="filipinoboxers",
    embeddings=embeddings
)

# Setup Groq LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# Define boxer names
KNOWN_ENTITIES = ["Manny Pacquiao", "Nonito Donaire"]

# Detect if user mentioned a known boxer
def extract_target_entity(query: str):
    for name in KNOWN_ENTITIES:
        if re.search(re.escape(name), query, re.IGNORECASE):
            return name
    return None

# Define prompt template
prompt = PromptTemplate.from_template(
    """Answer the question based only on the context below.

Context:
{context}

Question: {question}"""
)

# Output parser
output_parser = StrOutputParser()

# Main loop
print("\n🥊 Filipino Boxers RAG Chatbot. Type 'exit' to quit.")
while True:
    query = input("\n👤 You: ").strip()
    if query.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    target = extract_target_entity(query)

    # Use filtered retriever if boxer is identified
    if target:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 8, "filter": {"athlete": target}}
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build and run chain
    chain = prompt | llm | output_parser
    answer = chain.invoke({"context": context, "question": query})

    print("\n🤖 Answer:", answer)

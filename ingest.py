import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Load .env variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Target Wikipedia pages
PAGES = ["Manny Pacquiao", "Nonito Donaire"]

# Load and tag documents
all_docs = []
for name in PAGES:
    print(f"[+] Loading: {name}")
    try:
        docs = WikipediaLoader(query=name, load_max_docs=1).load()
        for doc in docs:
            doc.metadata["athlete"] = name
        all_docs.extend(docs)
    except Exception as e:
        print(f"[!] Failed to load {name}: {e}")

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)
print(f"[+] Total chunks: {len(split_docs)}")

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Reset collection
collection_name = "filipinoboxers"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print(f"🗑️ Deleted old collection: {collection_name}")

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print(f"✅ Created new collection: {collection_name}")

# Upload
qdrant = Qdrant.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=collection_name,
)

print("🚀 Ingestion complete!")

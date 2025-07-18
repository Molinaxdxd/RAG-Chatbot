from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

client.delete_collection(collection_name="filipinosportslegends")
print("✅ Deleted old collection")
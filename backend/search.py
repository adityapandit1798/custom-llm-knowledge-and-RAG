import faiss
import numpy as np
from openai import OpenAI
import os

D = 1536  # OpenAI embedding dimension
INDEX_PATH = "faiss.index"

# Load or create FAISS index
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(D)

def search(query: str, top_k: int = 5):
    """Search FAISS index for similar embeddings."""
    response = OpenAI.Embedding.create(input=query, model="text-embedding-ada-002")
    query_embedding = np.array(response["data"][0]["embedding"], dtype=np.float32)

    distances, indices = index.search(np.expand_dims(query_embedding, axis=0), top_k)
    return distances, indices

import sqlite3
import faiss
import openai
import numpy as np
import json
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Database and FAISS index paths
DB_PATH = "database.db"
INDEX_PATH = "faiss.index"

# Initialize FAISS index (1536 dimensions for OpenAI embeddings)
D = 1536  # OpenAI embedding dimension
index = faiss.IndexFlatL2(D)

# Database Connection
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Define Pydantic Model
class Document(BaseModel):
    url: str
    title: str
    summary: str
    content: str
    metadata: dict

# Load FAISS index from file (if exists)
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)

@app.on_event("startup")
def startup():
    """Create SQLite table on startup"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS site_pages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL,
                        title TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        embedding BLOB NOT NULL
                      )''')
    conn.commit()
    conn.close()

@app.post("/add_document/")
def add_document(doc: Document):
    """Add a document with embeddings to SQLite and FAISS."""
    try:
        # Generate embedding
        response = openai.Embedding.create(input=doc.content, model="text-embedding-ada-002")
        embedding = np.array(response["data"][0]["embedding"], dtype=np.float32)

        # Save to SQLite
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO site_pages (url, title, summary, content, metadata, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                       (doc.url, doc.title, doc.summary, doc.content, json.dumps(doc.metadata), embedding.tobytes()))
        conn.commit()
        conn.close()

        # Add to FAISS index
        index.add(np.expand_dims(embedding, axis=0))
        faiss.write_index(index, INDEX_PATH)  # ✅ Save FAISS index to file

        return {"message": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
def search(query: str, top_k: int = 5):
    """Search for similar documents using FAISS."""
    try:
        # Generate query embedding
        response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
        query_embedding = np.array(response["data"][0]["embedding"], dtype=np.float32)

        # Perform FAISS search
        distances, indices = index.search(np.expand_dims(query_embedding, axis=0), top_k)

        # Fetch results from SQLite
        conn = get_db()
        cursor = conn.cursor()
        results = []
        for i in indices[0]:
            cursor.execute("SELECT * FROM site_pages WHERE id=?", (i+1,))
            row = cursor.fetchone()
            if row:
                results.append(dict(row))
        conn.close()

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

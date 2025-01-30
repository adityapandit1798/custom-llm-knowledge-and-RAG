import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import SearchRequest, Document
from search import search_documents, add_document

app = FastAPI()

# Enable CORS (Frontend Communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
def search_api(request: SearchRequest):
    """Search API to retrieve similar documents."""
    results = search_documents(request.query, top_k=5)
    return {"results": results}

@app.post("/add")
def add_api(doc: Document):
    """API to add a new document."""
    add_document(doc.title, doc.content, doc.url)
    return {"message": "Document added successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

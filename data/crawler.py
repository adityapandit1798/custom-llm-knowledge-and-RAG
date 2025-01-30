import os
import json
import asyncio
import requests
import sqlite3
import faiss
import numpy as np
from xml.etree import ElementTree
from typing import List, Dict, Any
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# SQLite DB & FAISS Index paths
DB_PATH = "database.db"
INDEX_PATH = "faiss.index"

# Connect to SQLite
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create table for documents if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    url TEXT,
    embedding BLOB
)
""")
conn.commit()

# Load or create FAISS index
try:
    index = faiss.read_index(INDEX_PATH)
except:
    index = faiss.IndexFlatL2(384)  # Vector dimension = 384

# ✅ Chunking Function
def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Splits text into smaller chunks with context-aware boundaries."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        last_break = max(chunk.rfind('```'), chunk.rfind('\n\n'), chunk.rfind('. '))
        if last_break > chunk_size * 0.3:
            end = start + last_break + 1

        chunks.append(text[start:end].strip())
        start = max(start + 1, end)

    return chunks

# ✅ Extract Title & Summary
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extracts title & summary using GPT."""
    system_prompt = """Extract a concise title and summary from this text. 
    Return a JSON with 'title' and 'summary' keys."""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": f"URL: {url}\n\n{chunk[:1000]}..."}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting title/summary: {e}")
        return {"title": "Unknown", "summary": "No summary available"}

# ✅ Get Embedding
async def get_embedding(text: str) -> List[float]:
    """Returns embedding vector from OpenAI."""
    return embedding_model.encode(text).tolist()

# ✅ Process a Document Chunk
async def process_chunk(chunk: str, chunk_number: int, url: str):
    """Processes a document chunk & stores it in SQLite & FAISS."""
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)

    cursor.execute("INSERT INTO documents (title, content, url, embedding) VALUES (?, ?, ?, ?)",
                   (extracted["title"], chunk, url, json.dumps(embedding)))
    conn.commit()

    # Update FAISS Index
    index.add(np.array([embedding], dtype=np.float32))
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ Stored chunk {chunk_number} - {extracted['title']}")

# ✅ Process & Store Full Document
async def process_and_store_document(url: str, markdown: str):
    """Chunks a document & stores each chunk in parallel."""
    chunks = chunk_text(markdown)
    await asyncio.gather(*[process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)])

# ✅ Crawl URLs in Parallel
async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawls multiple pages in parallel."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(url=url, config=crawl_config, session_id="session1")
                if result.success:
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"❌ Failed: {url} - {result.error_message}")

        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

# ✅ Get URLs from Sitemap
def get_pydantic_ai_docs_urls() -> List[str]:
    """Fetches URLs from Pydantic AI documentation sitemap."""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        return [loc.text for loc in root.findall('.//ns:loc', namespace)]
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

# ✅ Main Entry Point
async def main():
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("⚠️ No URLs found!")
        return
    print(f"🌍 Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())

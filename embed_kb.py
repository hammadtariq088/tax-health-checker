"""
Standalone script to embed knowledge base files into pgvector using Groq API.

Usage:
    python embed_kb.py             # Truncates existing data, re-embeds everything
    python embed_kb.py --no-clear  # Skips truncation (only inserts new chunks)

Run this once before starting the server, or whenever knowledge_base/ files change.
The server (main.py) starts instantly without waiting for embeddings.
"""

import os
import re
import json
import glob
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import psycopg2
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("embed_kb")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY:
    raise SystemExit("FATAL: OPENAI_API_KEY not set in .env")
if not DATABASE_URL:
    raise SystemExit("FATAL: DATABASE_URL not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 0
EMBEDDING_DIMS = 1536
EMBED_BATCH_SIZE = 50
EMBED_SLEEP = 0.5
MAX_EMBED_RETRIES = 5

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_db():
    return psycopg2.connect(DATABASE_URL)


def init_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables WHERE table_name = 'knowledge_chunks'
        )
    """)
    exists = cur.fetchone()[0]

    if exists:
        cur.execute("""
            SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
            WHERE c.relname = 'knowledge_chunks'
              AND a.attname = 'embedding'
              AND a.attnum > 0
        """)
        row = cur.fetchone()
        if row:
            dim_match = re.search(r'\((\d+)\)', row[0])
            existing_dims = int(dim_match.group(1)) if dim_match else 0
            if existing_dims != EMBEDDING_DIMS:
                logger.warning(f"Dimension mismatch: table has {existing_dims}, model uses {EMBEDDING_DIMS}. Recreating...")
                cur.execute("DROP TABLE IF EXISTS knowledge_chunks CASCADE")
                exists = False

    if not exists:
        cur.execute("""
            CREATE TABLE knowledge_chunks (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                source_file VARCHAR(255),
                chunk_index INTEGER,
                embedding vector(%s)
            )
        """, (EMBEDDING_DIMS,))
        if EMBEDDING_DIMS <= 2000:
            cur.execute("""
                CREATE INDEX idx_knowledge_embedding
                ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """)
            logger.info("Created ivfflat index")
        else:
            logger.info(f"Skipping ivfflat index ({EMBEDDING_DIMS} dims > 2000)")

    conn.commit()
    cur.close()
    conn.close()
    logger.info("pgvector table ready")


def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        if end < text_len:
            period = text.rfind(".", start, end)
            newline = text.rfind("\n", start, end)
            split_at = max(period, newline)
            if split_at > start + CHUNK_SIZE // 2:
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP if end < text_len else text_len
    logger.info(f"  -> {len(chunks)} chunks")
    return chunks


def load_single_file(filepath: str) -> Optional[str]:
    try:
        ext = Path(filepath).suffix.lower()
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list):
                texts = []
                for item in data:
                    if isinstance(item, dict):
                        for key in ["text", "content", "message", "response"]:
                            if key in item and isinstance(item[key], str):
                                texts.append(item[key])
                                break
                        if "messages" in item and isinstance(item["messages"], list):
                            for msg in item["messages"]:
                                if isinstance(msg, dict) and "content" in msg:
                                    texts.append(str(msg["content"]))
                return "\n".join(texts)
            elif isinstance(data, dict):
                return json.dumps(data, indent=2)
            else:
                return str(data)
        else:
            logger.warning(f"Unsupported file type: {filepath}")
            return None
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return None


def embed_batch(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMS,
    )
    data = response.data
    return [data[i].embedding for i in range(len(texts))]


def embed_with_retry(texts: List[str]) -> List[List[float]]:
    last_error = None
    for attempt in range(MAX_EMBED_RETRIES):
        try:
            return embed_batch(texts)
        except Exception as e:
            last_error = e
            logger.warning(f"  Embed error (attempt {attempt + 1}/{MAX_EMBED_RETRIES}): {str(e)[:100]}")
            if attempt < MAX_EMBED_RETRIES - 1:
                time.sleep(min(10, 2 ** attempt))
            else:
                break
    logger.error(f"  All retries exhausted for batch")
    raise last_error


def embed_knowledge_base(clear: bool = True):
    kb_path = Path(KNOWLEDGE_BASE_DIR)
    if not kb_path.exists():
        logger.error(f"Directory {KNOWLEDGE_BASE_DIR} not found")
        return 0

    files = sorted(glob.glob(str(kb_path / "*.txt")) + glob.glob(str(kb_path / "*.json")))
    if not files:
        logger.warning(f"No .txt or .json files found in {KNOWLEDGE_BASE_DIR}")
        return 0

    logger.info(f"Found {len(files)} knowledge base files")

    if clear:
        logger.info("Truncating existing chunks...")
        conn = get_db()
        cur = conn.cursor()
        cur.execute("TRUNCATE knowledge_chunks")
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Existing chunks cleared")

    total_embedded = 0
    for filepath in files:
        filename = Path(filepath).name
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {filename}")
        logger.info(f"{'='*60}")

        content = load_single_file(filepath)
        if not content:
            logger.warning(f"  Skipping (no content extracted)")
            continue

        chunks = chunk_text(content)
        if not chunks:
            logger.warning(f"  Skipping (no chunks after splitting)")
            continue

        conn = get_db()
        cur = conn.cursor()
        batch_count = 0
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            batch_count += 1
            logger.info(f"  Batch {batch_count}: {len(batch)} chunks...")

            try:
                embeddings = embed_with_retry(batch)
            except Exception as e:
                logger.error(f"  Failed batch {batch_count} for {filename}: {e}")
                continue

            values = [
                (batch[j], filename, total_embedded + i + j, embeddings[j])
                for j in range(len(batch))
            ]
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO knowledge_chunks (chunk_text, source_file, chunk_index, embedding) VALUES %s",
                values,
                template="(%s, %s, %s, %s::vector)",
            )
            conn.commit()
            logger.info(f"  Batch {batch_count} stored ({len(batch)} chunks)")

            time.sleep(EMBED_SLEEP)

        cur.close()
        conn.close()
        total_embedded += len(chunks)
        logger.info(f"  Done: {len(chunks)} chunks embedded from {filename}")

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {total_embedded} total chunks embedded across {len(files)} files")
    logger.info(f"{'='*60}")
    return total_embedded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed knowledge base files into pgvector")
    parser.add_argument("--no-clear", action="store_true", help="Skip truncating existing chunks")
    args = parser.parse_args()

    logger.info("Starting knowledge base embedding...")
    init_table()
    embed_knowledge_base(clear=not args.no_clear)

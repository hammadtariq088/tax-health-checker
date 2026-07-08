import os
import re
import json
import glob
import io
import time
import random
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import pdfplumber
import google.generativeai as genai
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. AI features will use fallback responses.")
if not DATABASE_URL:
    logger.warning("DATABASE_URL not set. pgvector features will be unavailable.")

KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = None
EMBEDDING_DIMS = 768
EMBED_BATCH_SIZE = 50
EMBED_RATE_LIMIT_SLEEP = 1.5
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
TOP_K_CHUNKS = 8
MAX_CHUNK_CHARS = 6000

genai.configure(api_key=GEMINI_API_KEY)

_kb_loaded = False
_kb_loading = False
_kb_load_progress = {"total_chunks": 0, "embedded_chunks": 0, "status": "not_started"}


def resolve_embedding_model():
    global EMBEDDING_MODEL, EMBEDDING_DIMS
    try:
        available = [m.name for m in genai.list_models() if "embedContent" in m.supported_generation_methods]
        logger.info(f"Available embedding models: {available}")
    except Exception as e:
        logger.warning(f"Could not list models: {e}")
        available = []

    candidates = available or [
        "models/gemini-embedding-001",
        "models/gemini-embedding-2-preview",
        "models/gemini-embedding-2",
        "models/text-embedding-004",
        "models/embedding-001",
    ]

    for model in candidates:
        try:
            result = genai.embed_content(model=model, content="test")
            EMBEDDING_MODEL = model
            EMBEDDING_DIMS = len(result["embedding"])
            logger.info(f"Using embedding model: {model} ({EMBEDDING_DIMS} dims)")
            return
        except Exception as e:
            logger.warning(f"Embedding model {model} not available: {e}")
    logger.error("No embedding model available!")


def get_db() -> psycopg2.extensions.connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg2.connect(DATABASE_URL)


def init_db():
    global EMBEDDING_MODEL, EMBEDDING_DIMS
    resolve_embedding_model()
    if not DATABASE_URL:
        logger.error("Cannot initialize database: DATABASE_URL not set")
        return
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'knowledge_chunks'
            )
        """)
        table_exists = cur.fetchone()[0]

        if table_exists:
            cur.execute("""
                SELECT pg_typeof(embedding)::text
                FROM knowledge_chunks
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                dim_match = re.search(r'\((\d+)\)', row[0])
                existing_dims = int(dim_match.group(1)) if dim_match else 0
                if existing_dims != EMBEDDING_DIMS:
                    logger.warning(f"Embedding dimension changed from {existing_dims} to {EMBEDDING_DIMS}. Recreating table...")
                    cur.execute("DROP TABLE IF EXISTS knowledge_chunks CASCADE")
                    table_exists = False

        if not table_exists:
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
                    ON knowledge_chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                logger.info("Created ivfflat index on embedding column")
            else:
                logger.info(f"Skipping ivfflat index ({EMBEDDING_DIMS} dims > 2000 limit). Exact search will be used.")

        conn.commit()
        cur.close()
        conn.close()
        logger.info("pgvector table and index ready")
    except Exception as e:
        logger.error(f"Failed to initialize pgvector: {e}")


def count_chunks() -> int:
    if not DATABASE_URL:
        return 0
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM knowledge_chunks")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Failed to count chunks: {e}")
        return 0


def get_embedding(text: str) -> List[float]:
    if not EMBEDDING_MODEL:
        raise RuntimeError("No embedding model available. Check GEMINI_API_KEY.")
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
    )
    return result["embedding"]


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    if not EMBEDDING_MODEL:
        raise RuntimeError("No embedding model available. Check GEMINI_API_KEY.")
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
    )
    return result["embedding"]


def _parse_retry_delay(error_str: str) -> Optional[float]:
    match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', error_str)
    if match:
        return float(match.group(1))
    match = re.search(r'retry_delay\s*\{[^}]*seconds:\s*(\d+)', error_str)
    if match:
        return float(match.group(1))
    return None


def embed_with_retry(texts: List[str]) -> List[List[float]]:
    max_attempts = 20
    for attempt in range(max_attempts):
        try:
            return get_embeddings_batch(texts)
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Embedding API error (attempt {attempt + 1}/{max_attempts}): {error_str[:120]}")
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str:
                suggested_delay = _parse_retry_delay(error_str)
                delay = suggested_delay or min(60, RETRY_BASE_DELAY * (2 ** (attempt // 2)) + random.uniform(0, 2))
                logger.info(f"Quota limited. Waiting {delay:.0f}s before retry...")
                time.sleep(delay)
            elif attempt < max_attempts - 1:
                delay = min(30, RETRY_BASE_DELAY * (2 ** attempt))
                time.sleep(delay)
            else:
                break
    logger.error(f"All embedding retries exhausted after {max_attempts} attempts: {last_error}")
    raise last_error


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        full_text = "\n".join(text_parts).strip()
        if not full_text:
            raise ValueError("No extractable text found (scanned or empty PDF).")
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        return full_text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract text from this PDF. Please ensure it is a text-based PDF (not scanned). Error: {str(e)}",
        )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            split_at = max(last_period, last_newline)
            if split_at > start + chunk_size // 2:
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < text_len else text_len
    logger.info(f"Split text into {len(chunks)} chunks")
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


def load_knowledge_base() -> int:
    if not DATABASE_URL:
        logger.error("DATABASE_URL not configured. Cannot load knowledge base.")
        return 0

    kb_path = Path(KNOWLEDGE_BASE_DIR)
    if not kb_path.exists():
        kb_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created knowledge_base directory at {kb_path}")
        return 0

    existing = count_chunks()
    if existing > 0:
        logger.info(f"Knowledge base already loaded ({existing} chunks). Skipping reload.")
        return existing

    files = sorted(glob.glob(str(kb_path / "*.txt")) + glob.glob(str(kb_path / "*.json")))
    if not files:
        logger.warning(f"No knowledge base files found in {KNOWLEDGE_BASE_DIR}")
        return 0

    total_chunks = 0
    for filepath in files:
        content = load_single_file(filepath)
        if not content:
            continue
        chunks = chunk_text(content)
        filename = Path(filepath).name

        conn = get_db()
        cur = conn.cursor()
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            try:
                embeddings = embed_with_retry(batch)
            except Exception as e:
                logger.error(f"Failed to embed batch from {filename}: {e}")
                continue
            values = []
            for j, chunk_text_val in enumerate(batch):
                values.append((
                    chunk_text_val,
                    filename,
                    total_chunks + i + j,
                    embeddings[j],
                ))
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO knowledge_chunks (chunk_text, source_file, chunk_index, embedding) VALUES %s",
                values,
                template="(%s, %s, %s, %s::vector)",
            )
            conn.commit()
            logger.info(f"  Embedded and stored batch {i // EMBED_BATCH_SIZE + 1} ({len(batch)} chunks) from {filename}")
            time.sleep(EMBED_RATE_LIMIT_SLEEP)

        cur.close()
        conn.close()
        total_chunks += len(chunks)
        logger.info(f"Loaded {len(chunks)} chunks from {filename}")

    logger.info(f"Total: {total_chunks} chunks loaded from {len(files)} files")
    return total_chunks


def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_CHUNKS) -> List[str]:
    if not DATABASE_URL:
        logger.error("DATABASE_URL not configured. Cannot retrieve.")
        return []

    try:
        logger.info("Embedding query for retrieval...")
        query_embedding = get_embedding(query)

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT chunk_text, 1 - (embedding <=> %s::vector) AS similarity
            FROM knowledge_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if rows:
            logger.info(f"Retrieved {len(rows)} relevant chunks (top similarity: {rows[0][1]:.4f})")
            return [row[0] for row in rows]
        else:
            logger.info("No relevant chunks found in knowledge base")
            return []
    except Exception as e:
        logger.error(f"pgvector query failed: {e}")
        return []


def build_strict_prompt(tax_return_text: str, chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(chunks) if chunks else "No knowledge base chunks available."

    if len(context) > MAX_CHUNK_CHARS:
        context = context[:MAX_CHUNK_CHARS] + "\n...[truncated]"

    prompt = f"""You are a tax health checker for Tax Support Hub. Follow these rules STRICTLY:

1. ONLY use the provided knowledge base conversation chunks below. DO NOT use any external tax knowledge or your own training data.
2. If the knowledge base chunks contain NO relevant information about an issue, mark that area as "requires manual review".
3. Analyze the tax return content provided by the user.
4. Provide a helpful, non-technical assessment.

KNOWLEDGE BASE CHUNKS (the ONLY source of tax knowledge):
{context}

TAX RETURN CONTENT TO ANALYZE:
{tax_return_text}

Respond with valid JSON only (no markdown, no code fences, no extra text). Use this exact structure:
{{
  "overall_health": "green",
  "overall_summary": "One sentence summary of the overall tax return health.",
  "key_areas": [
    {{
      "area": "Name of the area",
      "status": "green",
      "explanation": "Simple, non-technical explanation for the client."
    }}
  ],
  "next_step": "We recommend scheduling a free one-to-one review with a Tax Support Hub professional to discuss your tax return in detail."
}}

Rules for status values:
- "green": No issues found, looks good.
- "yellow": Minor concerns or items that need attention.
- "red": Significant issues or high-risk areas that require immediate attention.

Include exactly 5 key areas in the key_areas array. Make explanations short and client-friendly (no jargon).
If no knowledge base context is available, set overall_health to "yellow" and explain that a manual review is needed.
"""
    return prompt


def call_gemini_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_google_gemini_api_key_here":
        return json.dumps({
            "overall_health": "yellow",
            "overall_summary": "AI service not configured. Please set up your Gemini API key.",
            "key_areas": [
                {
                    "area": "Configuration Required",
                    "status": "yellow",
                    "explanation": "The AI analysis service needs a valid Gemini API key to function. Please contact the site administrator."
                }
            ],
            "next_step": "Please set up your Gemini API key in the .env file and restart the server."
        })

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        }
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Gemini API (attempt {attempt + 1}/{max_retries})")
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            else:
                raise ValueError("Empty response from Gemini")
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.warning(f"Gemini API error (attempt {attempt + 1}): {error_str}")

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Rate limited. Retrying in {delay:.2f}s...")
                time.sleep(delay)
                continue
            elif "SAFETY" in error_str.upper() or "BLOCKED" in error_str.upper():
                return json.dumps({
                    "overall_health": "yellow",
                    "overall_summary": "The AI analysis was blocked by safety filters. A manual review is recommended.",
                    "key_areas": [
                        {
                            "area": "AI Safety Filter",
                            "status": "yellow",
                            "explanation": "The content could not be analyzed by AI. A tax professional should review this return manually."
                        }
                    ],
                    "next_step": "Please schedule a free one-to-one review with Tax Support Hub for a manual assessment."
                })
            else:
                if attempt < max_retries - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
                break

    logger.error(f"All Gemini API retries exhausted. Last error: {last_error}")
    return json.dumps({
        "overall_health": "yellow",
        "overall_summary": "The AI service is temporarily unavailable. A manual review is recommended.",
        "key_areas": [
            {
                "area": "Service Temporarily Unavailable",
                "status": "yellow",
                "explanation": "The analysis service could not be reached. Please try again later or schedule a manual review."
            }
        ],
        "next_step": "Please try again or schedule a free one-to-one review with Tax Support Hub."
    })


def parse_ai_response(response_text: str) -> dict:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = None
        else:
            result = None

    if result is None:
        logger.warning("Failed to parse AI response as JSON, using fallback")
        return {
            "overall_health": "yellow",
            "overall_summary": "We received an unparseable response from the analysis. A manual review is recommended.",
            "key_areas": [
                {
                    "area": "Analysis Response Issue",
                    "status": "yellow",
                    "explanation": "The AI returned a response that could not be processed. A tax professional should review this return."
                }
            ],
            "next_step": "Please schedule a free one-to-one review with Tax Support Hub for a thorough assessment."
        }

    if "key_areas" not in result or not isinstance(result["key_areas"], list):
        result["key_areas"] = [
            {
                "area": "General Assessment",
                "status": result.get("overall_health", "yellow"),
                "explanation": "Your tax return has been reviewed. Please see the overall summary above."
            }
        ]

    for area in result["key_areas"]:
        if "status" not in area:
            area["status"] = "yellow"
        if "area" not in area:
            area["area"] = "General"
        if "explanation" not in area:
            area["explanation"] = "This area requires further review by a tax professional."

    return result


# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

def load_knowledge_base_background():
    global _kb_loaded, _kb_loading, _kb_load_progress
    try:
        _kb_loading = True
        _kb_load_progress["status"] = "waiting_for_quota"
        logger.info("Waiting 60s for API quota to reset before embedding...")
        time.sleep(60)
        count = load_knowledge_base()
        _kb_loaded = True
        _kb_load_progress["status"] = "completed"
        _kb_load_progress["total_chunks"] = count
        logger.info(f"Knowledge base loaded: {count} chunks")
    except Exception as e:
        logger.error(f"Knowledge base loading failed: {e}")
        _kb_load_progress["status"] = f"failed: {e}"
        _kb_loaded = True
    finally:
        _kb_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _kb_loaded
    logger.info("Starting Tax Health Checker...")
    init_db()
    if not _kb_loaded:
        import threading
        t = threading.Thread(target=load_knowledge_base_background, daemon=True)
        t.start()
        logger.info("Knowledge base loading started in background. Server is ready immediately.")
    yield


app = FastAPI(
    title="Tax Health Checker",
    description="AI-powered tax return health check tool for Tax Support Hub",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = [
    "https://taxsupporthub.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    global _kb_loading, _kb_loaded, _kb_load_progress
    return {
        "status": "healthy",
        "knowledge_base_chunks": count_chunks(),
        "kb_loading": _kb_loading,
        "kb_loaded": _kb_loaded,
        "kb_progress": _kb_load_progress,
        "gemini_configured": GEMINI_API_KEY is not None and GEMINI_API_KEY != "your_google_gemini_api_key_here",
        "database_configured": DATABASE_URL is not None,
    }


@app.get("/api/kb-status")
def kb_status():
    global _kb_loading, _kb_loaded, _kb_load_progress
    chunks = count_chunks()
    if _kb_loaded:
        return {"ready": True, "loading": False, "chunks": chunks, "message": f"Knowledge base ready ({chunks} chunks)"}
    if _kb_loading:
        return {"ready": False, "loading": True, "chunks": chunks, "message": "Knowledge base is loading in background..."}
    return {"ready": False, "loading": False, "chunks": chunks, "message": "Knowledge base not yet loaded"}


@app.post("/api/reload-knowledge-base")
def reload_knowledge_base():
    global _kb_loaded, _kb_loading
    if DATABASE_URL:
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("TRUNCATE knowledge_chunks")
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Cleared existing knowledge base chunks")
        except Exception as e:
            logger.warning(f"Could not clear table: {e}")

    _kb_loaded = False
    _kb_loading = True
    import threading
    t = threading.Thread(target=load_knowledge_base_background, daemon=True)
    t.start()
    return {
        "success": True,
        "message": "Knowledge base reload started in background.",
    }


@app.post("/api/health-check")
async def health_check_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 50 MB limit.",
        )

    tax_return_text = extract_text_from_pdf(file_bytes)
    chunks = retrieve_relevant_chunks(tax_return_text)

    if not chunks:
        logger.info("No relevant chunks found. Returning manual review response.")
        return {
            "overall_health": "yellow",
            "overall_summary": "Your tax return has been reviewed against our knowledge base, but we need a tax professional to look at it closely.",
            "key_areas": [
                {
                    "area": "Knowledge Base Coverage",
                    "status": "yellow",
                    "explanation": "Our automated system could not find matching tax scenarios in our knowledge base for your specific return. This is not unusual — every tax return is unique."
                },
                {
                    "area": "Completeness Check",
                    "status": "yellow",
                    "explanation": "We were unable to verify if all required sections of your tax return are complete."
                },
                {
                    "area": "Compliance Review",
                    "status": "yellow",
                    "explanation": "Automated compliance checking could not be completed due to insufficient reference material."
                },
                {
                    "area": "Deduction & Credit Analysis",
                    "status": "yellow",
                    "explanation": "We could not automatically verify if all eligible deductions and credits have been claimed."
                },
                {
                    "area": "Risk Assessment",
                    "status": "yellow",
                    "explanation": "A full risk assessment requires manual review by a qualified tax professional."
                }
            ],
            "next_step": "Schedule a free one-to-one review with a Tax Support Hub professional to get a thorough analysis of your tax return."
        }

    prompt = build_strict_prompt(tax_return_text, chunks)
    ai_response = call_gemini_with_retry(prompt)
    result = parse_ai_response(ai_response)

    return result


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Frontend not found. Please ensure templates/index.html exists.</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

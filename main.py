import os
import re
import json
import io
import time
import random
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import pdfplumber
import psycopg2
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

class LeadCapture(BaseModel):
    email: str
    phone: str


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. AI features will use fallback responses.")
if not DATABASE_URL:
    logger.warning("DATABASE_URL not set. pgvector features will be unavailable.")

EMBEDDING_DIMS = 1536
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
TOP_K_CHUNKS = 8
MAX_CHUNK_CHARS = 6000

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def get_db() -> psycopg2.extensions.connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg2.connect(DATABASE_URL)


def init_db():
    if not DATABASE_URL:
        logger.error("Cannot initialize database: DATABASE_URL not set")
        return
    if not OPENAI_API_KEY:
        logger.error("Cannot initialize: OPENAI_API_KEY not set")
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

        cur.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                phone VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

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
    if not OPENAI_API_KEY or not client:
        raise RuntimeError("No embedding API key configured. Check OPENAI_API_KEY.")
    try:
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMS,
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding error: {str(e)}")


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


def call_openai_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    if not OPENAI_API_KEY or not client:
        return json.dumps({
            "overall_health": "yellow",
            "overall_summary": "AI service not configured. Please set up your OpenAI API key.",
            "key_areas": [
                {
                    "area": "Configuration Required",
                    "status": "yellow",
                    "explanation": "The AI analysis service needs a valid OpenAI API key to function. Please contact the site administrator."
                }
            ],
            "next_step": "Please set up your OpenAI API key in the .env file and restart the server."
        })

    system_prompt = (
        "You are a tax health checker for Tax Support Hub. You respond only with valid JSON "
        "following the exact structure requested by the user, using only the provided knowledge base."
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
            else:
                raise ValueError("Empty response from OpenAI")
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.warning(f"OpenAI API error (attempt {attempt + 1}): {error_str}")

            if "RateLimitError" in type(e).__name__ or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Rate limited. Retrying in {delay:.2f}s...")
                time.sleep(delay)
                continue
            elif "SAFETY" in error_str.upper() or "BLOCKED" in error_str.upper() or "content_policy" in error_str.lower():
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

    logger.error(f"All OpenAI API retries exhausted. Last error: {last_error}")
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Tax Health Checker...")
    init_db()
    chunks = count_chunks()
    logger.info(f"Knowledge base: {chunks} chunks ready")
    yield


app = FastAPI(
    title="Tax Health Checker",
    description="AI-powered tax return health check tool for Tax Support Hub",
    version="1.2.0",
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
    chunks = count_chunks()
    return {
        "status": "healthy",
        "knowledge_base_chunks": chunks,
        "kb_ready": chunks > 0,
        "openai_configured": OPENAI_API_KEY is not None and OPENAI_API_KEY != "sk-your_openai_api_key_here",
        "embeddings_configured": OPENAI_API_KEY is not None,
        "database_configured": DATABASE_URL is not None,
    }


@app.get("/api/kb-status")
def kb_status():
    chunks = count_chunks()
    ready = chunks > 0
    return {"ready": ready, "chunks": chunks, "message": f"Knowledge base: {chunks} chunks"}


@app.post("/api/reload-knowledge-base")
def reload_knowledge_base():
    if DATABASE_URL:
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("TRUNCATE knowledge_chunks")
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Cleared existing knowledge base chunks")
            return {
                "success": True,
                "message": "Knowledge base cleared. Run `python embed_kb.py` to reload.",
            }
        except Exception as e:
            logger.warning(f"Could not clear table: {e}")
            return {
                "success": False,
                "message": f"Failed to clear: {e}",
            }
    return {
        "success": False,
        "message": "DATABASE_URL not configured",
    }


@app.post("/api/capture-lead")
def capture_lead(data: LeadCapture):
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO leads (email, phone) VALUES (%s, %s)",
            (data.email.strip(), data.phone.strip()),
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Lead captured: {data.email}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to capture lead: {e}")
        raise HTTPException(status_code=500, detail="Failed to save lead info")


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
    ai_response = call_openai_with_retry(prompt)
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

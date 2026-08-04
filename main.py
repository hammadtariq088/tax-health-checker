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
4. Provide a thorough, helpful, non-technical assessment.

KNOWLEDGE BASE CHUNKS (the ONLY source of tax knowledge):
{context}

TAX RETURN CONTENT TO ANALYZE:
{tax_return_text}

Respond with valid JSON only (no markdown, no code fences, no extra text). Use this exact structure:
{{
  "overall_health": "green",
  "overall_summary": "A detailed, comprehensive paragraph summarising the overall health of the tax return.",
  "key_areas": [
    {{
      "area": "Name of the area",
      "status": "green",
      "explanation": "A detailed, comprehensive explanation written in full paragraphs."
    }}
  ],
  "next_step": "We recommend scheduling a free one-to-one review with a Tax Support Hub professional to discuss your tax return in detail."
}}

Rules for status values:
- "green": No issues found, looks good.
- "yellow": Minor concerns or items that need attention.
- "red": Significant issues or high-risk areas that require immediate attention.

Writing requirements (VERY IMPORTANT):
- Write the overall_summary as a detailed paragraph of at least 3-4 complete sentences covering the overall state of the return.
- Write each key area explanation as a detailed, comprehensive paragraph of at least 4-6 complete sentences. Explain what was checked, what was found, what it means for the client, and any action that should be taken.
- Use proper paragraph form with complete sentences. Do NOT use bullet points, lists, headings or terse one-line answers.
- Keep the language client-friendly and non-technical (no jargon).
- Include exactly 5 key areas in the key_areas array.
- If no knowledge base context is available, set overall_health to "yellow" and explain that a manual review is needed.
"""
    return prompt


def call_openai_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    if not OPENAI_API_KEY or not client:
        return json.dumps({
            "overall_health": "yellow",
            "overall_summary": "The AI analysis service has not been configured with a valid OpenAI API key, so your tax return could not be reviewed by our automated assistant. This is a technical configuration step on our side and does not reflect on your tax return in any way. A manual review by one of our tax professionals will provide you with the same careful attention.",
            "key_areas": [
                {
                    "area": "Configuration Required",
                    "status": "yellow",
                    "explanation": "The AI analysis service needs a valid OpenAI API key to function, and this key has not yet been configured. Without it, the automated assistant cannot securely read or analyse your tax return. This is entirely a technical setup matter on our side and does not indicate any problem with the document you uploaded. Please contact the site administrator so the service can be enabled, and in the meantime you can still receive a thorough review from one of our tax professionals."
                }
            ],
            "next_step": "Please set up your OpenAI API key in the .env file and restart the server, or schedule a free one-to-one review with Tax Support Hub for a manual assessment."
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
                    "overall_summary": "Our automated analysis was unable to process this tax return because the content triggered the AI safety filter. This is a protective safeguard designed to keep your information secure, and it does not mean there is anything wrong with your return. A manual review by one of our tax professionals will give you the thorough assessment you need.",
                    "key_areas": [
                        {
                            "area": "AI Safety Filter",
                            "status": "yellow",
                            "explanation": "The automated system declined to analyse this document because its content was flagged by the safety filter. These filters are deliberately cautious to protect your privacy and to avoid generating guidance from unclear or restricted material. As a result, we cannot offer automated findings on this occasion. A qualified tax professional will review your return manually to check its completeness, its compliance with current rules, the deductions and credits that apply, and any potential risks."
                        }
                    ],
                    "next_step": "Please schedule a free one-to-one review with Tax Support Hub for a thorough manual assessment."
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
        "overall_summary": "The AI analysis service is temporarily unavailable, so we could not complete an automated review of your tax return at this moment. This is a temporary technical issue on our side and does not reflect on your tax return. Please try again shortly, or speak with one of our tax professionals for an immediate manual assessment.",
        "key_areas": [
            {
                "area": "Service Temporarily Unavailable",
                "status": "yellow",
                "explanation": "Our automated analysis service could not be reached after several attempts. This can happen during brief maintenance windows or periods of high demand. Your tax return has not been affected, and no data has been lost. Please retry the analysis in a few minutes, or contact Tax Support Hub to book a manual review so your return can still be assessed properly."
            }
        ],
        "next_step": "Please try the analysis again in a few minutes, or schedule a free one-to-one review with Tax Support Hub."
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
            "overall_summary": "Our automated assistant returned a response that could not be processed into a readable report. This is a rare technical issue and does not indicate a problem with your tax return. To make sure your return is still reviewed properly, we recommend a manual assessment by one of our tax professionals.",
            "key_areas": [
                {
                    "area": "Analysis Response Issue",
                    "status": "yellow",
                    "explanation": "The AI produced a response that our system could not interpret, so no automated findings could be generated. This occasionally happens when the analysis service returns information in an unexpected format. It does not affect the safety or status of your uploaded document. A tax professional should review this return manually to ensure every section is checked, including completeness, compliance with current rules, deductions and credits, and any potential risks."
                }
            ],
            "next_step": "Please schedule a free one-to-one review with Tax Support Hub for a thorough manual assessment."
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
            "overall_summary": "Your tax return was received successfully, but our automated assistant could not find matching scenarios in our knowledge base to compare it against. This is not unusual, as every tax return is unique, and it simply means that a detailed manual review by one of our tax professionals is the best next step for you.",
            "key_areas": [
                {
                    "area": "Knowledge Base Coverage",
                    "status": "yellow",
                    "explanation": "Our automated system could not find tax scenarios in our knowledge base that closely match the details in your specific return. This happens when a return is unique or contains uncommon arrangements, and it is not a cause for concern. It simply means the automated assistant could not draw a reliable comparison, so a human expert is better placed to give you accurate guidance. Your return will still be looked at carefully by a qualified professional who can give you the attention it deserves."
                },
                {
                    "area": "Completeness Check",
                    "status": "yellow",
                    "explanation": "We were unable to automatically verify that all required sections of your tax return are complete. Completeness covers things like your personal details, income statements, deductions and any supporting schedules. Because a full automated comparison could not be run, we cannot confirm at this stage whether anything is missing or outstanding. A tax professional will review every section of your return to make sure nothing has been overlooked."
                },
                {
                    "area": "Compliance Review",
                    "status": "yellow",
                    "explanation": "Automated compliance checking could not be completed because there was insufficient reference material to compare against your return. Compliance means making sure your return follows the current tax rules, including the correct treatment of income, expenses and reporting requirements. Without a reliable comparison, we prefer not to guess at your situation. A qualified professional will check your return against the current rules to help ensure it is fully compliant."
                },
                {
                    "area": "Deduction & Credit Analysis",
                    "status": "yellow",
                    "explanation": "We could not automatically verify whether all the deductions and tax credits you may be entitled to have been claimed on your return. Deductions reduce the tax you pay, and missing one could mean you pay more than necessary. Because this check requires careful interpretation of your individual circumstances, we recommend having a professional review your expenses and entitlements so that every benefit you are owed is claimed."
                },
                {
                    "area": "Risk Assessment",
                    "status": "yellow",
                    "explanation": "A full risk assessment involves checking for errors, inconsistencies or anything that could attract attention from the tax authority. This kind of review requires judgement and experience, which our automated assistant is not able to provide without a reliable knowledge base match. A qualified tax professional will examine your return for potential risks and help you address them before any issues arise."
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

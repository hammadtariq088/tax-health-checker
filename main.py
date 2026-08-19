import os
import re
import io
import time
import random
import logging
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
import pdfplumber
import psycopg2
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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
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

MIN_REPORT_CHARS = 1200

CONSISTENT_AREA_SECTIONS = []

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

        cur.execute("""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = 'knowledge_chunks'
              AND indexname = 'idx_knowledge_embedding'
        """)
        index_row = cur.fetchone()
        if index_row and "USING hnsw" in index_row[0]:
            logger.info("HNSW index already exists on embedding column")
        else:
            if index_row:
                logger.info("Replacing existing vector index with HNSW...")
                cur.execute("DROP INDEX IF EXISTS idx_knowledge_embedding")
            cur.execute("""
                CREATE INDEX idx_knowledge_embedding
                ON knowledge_chunks
                USING hnsw (embedding vector_cosine_ops)
            """)
            logger.info("Created HNSW index on embedding column")

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


def build_review_prompt(tax_return_text: str, chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(chunks) if chunks else "No knowledge base chunks available."

    if len(context) > MAX_CHUNK_CHARS:
        context = context[:MAX_CHUNK_CHARS] + "\n...[truncated]"

    prompt = f"""You are acting as a senior Pakistan income tax consultant reviewing the income tax return, wealth statement, tax computation and supporting information of an individual taxpayer from the perspective of potential FBR scrutiny, departmental inquiry, amendment proceedings, wealth reconciliation issues, unexplained income/assets, withholding mismatches and other material tax risks.

Use:

1. the documents provided for the taxpayer;
2. the Income Tax Ordinance, 2001 available in the knowledge base;
3. relevant tax knowledge, practical departmental experience, notice patterns, risk indicators and tax matters already available in the existing knowledge base and conversation history.

Do not invent legal provisions, facts, SROs, rules or case law. Where information is insufficient, identify the risk conditionally instead of assuming facts.

PRIMARY OBJECTIVE

Do not prepare a long-form tax review.

Your task is to identify only the six most important, genuine and value-adding Areas of Concern in the taxpayer's return.

The purpose is to provide the taxpayer with a concise diagnostic review explaining:

- what looks unusual, inconsistent or potentially problematic;
- why the matter may attract FBR attention;
- what type of inquiry, notice, explanation or documentation may potentially be required; and
- what practical action should be considered.

Do not generate observations merely for the sake of completing six points.

Only include issues that are reasonably supported by the taxpayer's information and are significant enough to deserve attention.

AREAS TO REVIEW

While selecting the most important concerns, review the return from the following perspectives:

- inconsistency between income and wealth;
- unexplained increase in assets;
- opening and closing wealth mismatch;
- possible exposure under section 111;
- unusual increase or decrease in bank balances;
- unidentified bank credits;
- assets acquired without an identifiable source;
- gifts, loans, inheritances or remittances with weak documentation;
- first-time appearance of material assets;
- property acquisition or disposal;
- foreign assets or foreign income;
- mismatch between IRIS withholding data and tax claimed;
- tax deducted but not claimed;
- tax claimed without sufficient support;
- incorrect tax regime or income classification;
- income incorrectly treated as exempt, final tax or separately taxed;
- omitted income;
- unusual movement compared with previous years;
- major decline in taxable income without corresponding commercial explanation;
- unusually low personal expenditure compared with income/assets/lifestyle;
- business capital not reconciling with personal wealth;
- investments/redemptions not properly reflected;
- sale proceeds confused with capital gains;
- liabilities introduced without identifiable supporting evidence;
- incorrect or weak wealth reconciliation;
- tax refund or credit anomalies;
- any matter likely to be visible to FBR through third-party information.

Also consider whether there is any major tax-saving, adjustable tax, refund or correction opportunity, but include it only if it is significant enough to rank among the six most important observations.

DEPARTMENTAL RISK LENS

Review the taxpayer's information from two perspectives simultaneously:

Taxpayer Perspective:
Is the return legally and factually defensible?

FBR Perspective:
If an officer reviews the return, what unusual item is most likely to trigger a question, notice, reconciliation request or documentary inquiry?

Particular emphasis should be placed on issues that can potentially be identified through:

- IRIS information;
- withholding statements;
- banking information;
- property records;
- vehicle information;
- investment information;
- employer information;
- other third-party data available to the tax authorities.

WEALTH STATEMENT REVIEW

Do not merely check whether the wealth reconciliation mathematically balances.

Assess whether the underlying sources of wealth are credible and supported.

Check whether:

- opening wealth agrees with the preceding year;
- asset additions have identifiable sources;
- disposal proceeds are correctly reflected;
- liabilities are genuine and supported;
- gifts/loans/remittances have documentary evidence;
- bank balances are consistent with available funds;
- personal expenses appear reasonable;
- investments are properly reconciled;
- assets disposed of have been removed;
- current-year acquisitions are properly disclosed.

Where appropriate, identify potential section 111 risk, but do not state conclusively that section 111 applies unless the facts support such a conclusion.

PRIOR-YEAR COMPARISON

Where previous returns are available, compare the current year with prior years.

Look particularly for material movement in:

- income;
- taxable income;
- exempt income;
- bank balances;
- investments;
- properties;
- vehicles;
- cash;
- foreign assets;
- liabilities;
- personal expenses;
- net wealth.

A large movement should not automatically be considered an error. Flag it only where the source or explanation is unclear, inconsistent or insufficiently documented.

REQUIRED OUTPUT FORMAT

The output must be short, client-friendly and highly focused.

Provide only 6 key Areas of Concern.

For each issue, use exactly the following style:

1. [Short Risk Heading]

Write one concise paragraph of approximately 5 lines.

The paragraph should naturally cover:

- what has been identified;
- why it is unusual or potentially risky;
- what FBR may potentially question or seek;
- the possible consequence or exposure; and
- the practical course of action.

Do not create separate subheadings such as:

- Observation;
- Risk;
- FBR Query;
- Recommendation;
- Legal Position.

Everything should be incorporated naturally into the same short paragraph.

Then continue:

2. [Short Risk Heading]

5 line paragraph.

Continue in the same format for a maximum of 6 issues.

IMPORTANT OUTPUT RULES

1. Provide only the six strongest observations.
2. Do not provide generic tax advice.
3. Do not mention minor or technical errors unless they can have a meaningful impact.
4. Do not manufacture risks just to reach six observations.
5. If only three or four genuine concerns exist, provide only those.
6. Every observation must arise from the taxpayer's actual information.
7. Keep each observation to approximately 5 lines only.
8. Use plain professional English that an individual taxpayer can understand.
9. Avoid excessive legal terminology.
10. Mention a legal provision only where it materially adds value.
11. Where the taxpayer has a defensible position, make that clear.
12. Where further information is needed, state what needs to be verified.
13. Prioritise matters capable of leading to:

- FBR inquiry;
- notice;
- documentary requisition;
- amendment;
- section 111 proceedings;
- additional tax exposure; or
- correction/revision of the return.

14. Focus on actionable risks, not academic observations.
15. The final result should read like a short professional tax diagnostic prepared for the taxpayer, not a detailed tax audit report.

FINAL INSTRUCTION

After reviewing all available information, rank the issues from most significant to least significant.

The final answer should contain only:

"Key Areas of Concern"

followed by the 6 numbered risk headings and their respective 5 line explanatory paragraphs.

Do not add an executive summary, conclusion, detailed tables, disclaimer or lengthy legal analysis unless specifically requested.

## DATABASE CONTROL (mandatory)

The knowledge base chunks below are your ONLY source of tax law and statutory knowledge. Before reaching any conclusion, search them, apply the law for the tax year of the uploaded return, and DO NOT apply later law retrospectively. Do not quote a rate, threshold, section, rule, notification or judgment unless verified from the chunks. Where the database does not provide a conclusive answer, state "Further legal verification is required; no definitive adverse conclusion should presently be drawn." NEVER invent statutory provisions, thresholds, rates, judicial principles or document requirements.

KNOWLEDGE BASE CHUNKS (the connected tax database - the ONLY source of tax law):
{context}

TAXPAYER INPUT (income tax return and related information extracted from the uploaded PDF):
{tax_return_text}

FINAL OUTPUT INSTRUCTION:
Respond ONLY with the "Key Areas of Concern" heading followed by the numbered risk observations. Do NOT use JSON. Do NOT wrap the report in markdown code fences. Do NOT include an Overall Assessment or executive summary. Start directly with "Key Areas of Concern" as the title, then list each numbered observation with its heading and approximately 5-line paragraph. If fewer than 6 genuine concerns exist, provide only those that are genuinely supported by the taxpayer's information. A one- or two-sentence report is NEVER acceptable.
"""
    return prompt


def build_fallback_report(
    summary: str,
    observations: List[dict],
) -> str:
    """Build a numbered-format fallback report for the Key Areas of Concern output.

    Each observation dict may carry 'heading' and 'text'. The result uses the
    same numbered format as the AI-generated report so the frontend renders
    it identically.
    """
    parts = []
    idx = 1
    for obs in observations:
        heading = (obs.get("heading") or "General Assessment").strip()
        text = (obs.get("text") or obs.get("observation") or "").strip()
        if not text:
            continue
        parts.append(f"{idx}. {heading}\n\n{text}")
        idx += 1

    if not parts:
        parts.append(
            "1. Manual Review Recommended\n\n"
            "A tax professional should manually review this return to confirm the "
            "position before filing, as automated analysis could not be completed."
        )

    return "Key Areas of Concern\n\n" + "\n\n".join(parts)


def call_openai_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    if not OPENAI_API_KEY or not client:
        return build_fallback_report(
            summary="",
            observations=[
                {
                    "heading": "Configuration Required",
                    "text": "The AI analysis service needs a valid OpenAI API key to function, and this key has not yet been configured. Without it, the automated assistant cannot securely read or analyse your tax return. This is entirely a technical setup matter on our side and does not indicate any problem with the document you uploaded. Please contact the site administrator so the service can be enabled, or schedule a free one-to-one review with Tax Support Hub for a manual assessment.",
                }
            ],
        )

    system_prompt = (
        "You are an expert Pakistan Income Tax Return Review, Risk Assessment and Preventive "
        "Compliance Assistant for Tax Support Hub. You perform a professional face-of-return "
        "tax risk review of individual income tax returns and wealth statements. You respond "
        "ONLY with the numbered Key Areas of Concern report defined in the user's instructions, "
        "using only the provided knowledge base as the source of tax law."
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
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
                return build_fallback_report(
                    summary="",
                    observations=[
                        {
                            "heading": "AI Safety Filter",
                            "text": "The automated system declined to analyse this document because its content was flagged by the safety filter. These filters are deliberately cautious to protect your privacy and to avoid generating guidance from unclear or restricted material. As a result, we cannot offer automated findings on this occasion, and a qualified tax professional should review this return manually to check its completeness, compliance, deductions, credits and any potential risks. Please schedule a free one-to-one review with Tax Support Hub for a thorough manual assessment.",
                        }
                    ],
                )
            else:
                if attempt < max_retries - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
                break

    logger.error(f"All OpenAI API retries exhausted. Last error: {last_error}")
    return build_fallback_report(
        summary="",
        observations=[
            {
                "heading": "Service Temporarily Unavailable",
                "text": "Our automated analysis service could not be reached after several attempts. This can happen during brief maintenance windows or periods of high demand. Your tax return has not been affected, and no data has been lost. Please retry the analysis in a few minutes, or contact Tax Support Hub to book a manual review.",
            }
        ],
    )


def ensure_full_report(report_text: str) -> str:
    """Guarantee a minimum-quality report for the Key Areas of Concern format.

    If the report has fewer than 2 numbered observations or is too short,
    return it as-is without padding. The new format does not use confirmed-
    consistent sections or an Overall Assessment.
    """
    numbered_count = len(re.findall(r'(?:^|\n)\s*\d+\.\s', report_text))
    if numbered_count >= 2 or len(report_text) >= MIN_REPORT_CHARS:
        return report_text

    return report_text


def clean_report_text(response_text: str) -> str:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:markdown|md|text)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    return ensure_full_report(cleaned)


def parse_ai_response(response_text: str) -> dict:
    return {"report": clean_report_text(response_text)}


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
    description="AI-powered tax return risk review tool for Tax Support Hub",
    version="2.0.0",
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
        return parse_ai_response(build_fallback_report(
            summary="",
            observations=[
                {
                    "heading": "Knowledge Base Coverage",
                    "text": "Our automated system could not find tax scenarios in our knowledge base that closely match the details in your specific return. This happens when a return is unique or contains uncommon arrangements, and it is not a cause for concern. It simply means the automated assistant could not draw a reliable comparison, so a human expert is better placed to give you accurate guidance. A qualified professional will review every section of your return, including completeness, compliance, deductions, credits and any potential risks, so that every benefit you are owed is claimed and nothing has been overlooked.",
                }
            ],
        ))

    prompt = build_review_prompt(tax_return_text, chunks)
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

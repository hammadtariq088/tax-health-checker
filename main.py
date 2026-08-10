import os
import re
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

RISK_LEVELS = ["High", "Medium", "Low"]

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

    prompt = f"""# MASTER PROMPT - PAKISTAN INDIVIDUAL INCOME TAX RETURN RISK REVIEW

## ROLE

Act as a highly experienced Pakistani income tax consultant specialising in individual income tax returns, wealth statements, wealth reconciliation, salary cases, business/trader cases, withholding taxes, property transactions, investments, foreign remittances, tax credits, and FBR proceedings.

Your job is NOT to summarize the uploaded return.

Your job is to perform a professional "face-of-return tax risk review" and identify only genuine potential areas of concern that could reasonably lead to an FBR query, verification, audit observation, amendment proceeding, unexplained income/asset issue, tax shortfall, or documentary challenge in the future.

Think like a senior Pakistani tax consultant reviewing a client's return before an FBR notice is received.

Apply the Income Tax Ordinance, 2001, relevant Finance Act amendments, Income Tax Rules, and other applicable provisions according to the TAX YEAR of the uploaded return. Never apply a later amendment retrospectively unless the law specifically requires it.

## CORE OBJECTIVE

Review the complete Return of Income, Wealth Statement, Wealth Reconciliation, tax computation, withholding tax details, tax credits, and all other available schedules TOGETHER.

Do not review each field independently.

Cross-check figures and disclosures against each other and identify:

* unusual relationships;
* material inconsistencies;
* missing corresponding disclosures;
* questionable sources of funds;
* unsupported wealth movements;
* possible incorrect tax treatment;
* withholding mismatches;
* computational issues;
* documentary weaknesses.

The final report must be understandable to an ordinary Pakistani taxpayer with little or no tax or accounting knowledge.

## MOST IMPORTANT RULE - NEVER FORCE A RISK

Do NOT create observations merely to make the report look comprehensive.

A large amount is NOT automatically a risk.

A property, investment, salary, business turnover, bank balance, cash balance, loan, gift, foreign asset, capital gain, tax credit or other transaction should not be flagged merely because its value is high.

There must be a genuine reason for concern, such as:

* inconsistency;
* unexplained source;
* unusual relationship;
* possible tax shortfall;
* missing corresponding income;
* questionable tax treatment;
* material documentary exposure.

If only 2 genuine concerns exist, report only 2.

If 6 genuine concerns exist, report 6.

If there are no meaningful concerns visible from the return, clearly state that no significant risk or inconsistency was identified from the information available.

Normally, the report should contain no more than 5-10 observations. Fewer than 5 is completely acceptable.

## ANALYTICAL REVIEW PROCESS

Before writing the report, silently perform the following checks.

## 1. RETURN OF INCOME VS WEALTH STATEMENT

Compare declared income with the movement in the taxpayer's wealth.

Determine whether increases in assets appear reasonably supported by:

* taxable income;
* exempt income;
* final/fixed tax income;
* gifts;
* inheritance;
* foreign remittances;
* loans;
* disposal of assets;
* other disclosed inflows.

Identify any material increase in wealth that does not appear adequately supported by the declared sources.

Do not assume that a mathematically balanced wealth reconciliation means everything is correct. Examine the substance of the amounts used to make it balance.

## 2. WEALTH RECONCILIATION

Review:

* opening net assets;
* closing net assets;
* increase/decrease in wealth;
* income declared;
* personal expenses;
* gifts;
* foreign remittances;
* loans;
* inheritance;
* other inflows;
* other outflows;
* unreconciled amount.

Pay particular attention to material figures appearing under vague descriptions such as "Others", "Other Inflows", "Other Assets", "Receivables", or similar generic categories.

A zero unreconciled amount is positive, but it does NOT automatically mean the wealth statement is risk-free.

## 3. CASH AND BANK POSITION

Compare cash in hand with:

* bank balances;
* annual income;
* business activity;
* personal expenses;
* total assets;
* previous-year cash, if available.

Flag unusually high cash in hand only where it appears financially or commercially difficult to justify.

Pay particular attention where cash appears to have increased significantly without an obvious economic reason or appears to have been used primarily as a balancing figure in the wealth statement.

Do not flag normal or immaterial cash balances.

## 4. GIFTS

Where a material gift appears, consider:

* size of the gift relative to income;
* size relative to net wealth;
* whether it is the main source of asset growth;
* identity of the donor;
* apparent financial capacity of the donor, where information is available;
* banking trail;
* supporting documentation.

Do not automatically describe a gift as taxable.

Explain the real concern in simple language.

Where appropriate, recommend maintaining the gift deed, banking trail, donor identification and evidence supporting the donor's financial capacity.

## 5. FOREIGN REMITTANCES AND FOREIGN ASSETS

Apply the law applicable to the relevant Tax Year.

Review:

* amount of foreign remittance;
* mode/channel through which it was received;
* available banking evidence;
* PRC or equivalent evidence where relevant;
* nature and source of funds;
* foreign assets;
* corresponding foreign income;
* consistency between foreign remittances, foreign assets and declared income.

Do not automatically treat every foreign remittance as exempt or taxable.

Identify the actual statutory, source-of-funds or documentary condition that creates the potential exposure.

## 6. SALARY CASES - MUST PERFORM AN INDEPENDENT CHECK

Whenever salary income is present, independently compute/review the salary tax using the rates applicable to that Tax Year.

Compare:

Declared Salary vs. Expected Salary Tax vs. Employer Withholding under Section 149 vs. Tax Claimed vs. Refund / Tax Payable.

Look for material differences that may indicate:

* omitted salary;
* bonus or arrears not properly reported;
* taxable benefits/perquisites not reflected;
* incorrect employer withholding;
* incorrect tax computation;
* excessive refund claim;
* short deduction of tax.

If salary and withholding appear reasonably consistent, do NOT manufacture an observation.

Only include salary in the final report where the analysis produces something useful for the taxpayer.

## 7. BUSINESS / TRADER CASES - PERFORM COMMERCIAL ANALYSIS

Where business or trading income exists, review:

* turnover/sales;
* gross profit;
* net profit;
* purchases;
* opening and closing stock;
* major expenses;
* withholding taxes linked with sales/receipts;
* debtors;
* creditors;
* business cash;
* business bank accounts;
* business assets.

Check whether withholding information indicates business receipts materially higher than the turnover declared in the return.

Compare turnover, purchases, stock, gross profit and expenses to determine whether they make commercial sense together.

Identify material expenses parked under headings such as "Miscellaneous Expenses", "Other Expenses", "General Expenses" or "Administrative Expenses" where insufficient classification may create difficulty in establishing their nature or allowability.

Consider whether substantial sales exist without corresponding purchases, expenses, stock movements or commercially reasonable profit.

Do NOT flag a business simply because its gross profit or net profit is high or low. There must be a meaningful inconsistency visible from the return.

## 8. WITHHOLDING TAX - USE IT AS A TRANSACTION DETECTOR

Do not review withholding tax merely as a tax credit.

Ask: "What underlying transaction must have occurred for this withholding tax to arise?"

Then check whether the corresponding income, receipt, asset or transaction appears elsewhere in the return.

Examples include:

Salary withholding -> compare with salary income.
Sales/contract withholding -> compare with declared business turnover.
Property purchase withholding -> compare with property additions.
Property sale withholding -> compare with property disposal and capital gain.
Profit-on-debt withholding -> compare with bank/investment income.
Dividend withholding -> compare with dividend income and investments.

A material mismatch should receive HIGH priority.

## 9. PROPERTY TRANSACTIONS

For property purchases and disposals, cross-check:

* property appearing in the wealth statement;
* acquisition/disposal value;
* applicable withholding tax;
* source of investment;
* financing;
* loans;
* capital gain, where applicable;
* corresponding movement in wealth.

Only report meaningful inconsistencies, incorrect tax treatment or source-of-funds concerns.

## 10. LOANS, RECEIVABLES AND LIABILITIES

Review material loans, advances, receivables and liabilities.

Consider:

* size relative to income and wealth;
* nature of the transaction;
* counterparty, where available;
* source of funds;
* movement from the previous year;
* whether the transaction makes financial sense;
* whether documentary support would ordinarily be expected.

Do not flag genuine bank financing merely because the amount is large where the related asset and financing appear consistent.

## 11. INVESTMENTS AND INVESTMENT INCOME

Where significant investments exist, determine whether related income appears where reasonably expected, including:

* profit on debt;
* dividends;
* capital gains;
* mutual fund income.

Do NOT assume every investment must generate taxable income every year.

Only raise an observation where the information in the return provides a reasonable basis for concern.

## 12. TAX CREDITS AND REFUNDS

Review material:

* tax credits;
* donations;
* pension contributions;
* eligible investments;
* foreign tax credits;
* refundable withholding taxes.

Check whether the amount claimed appears consistent with the underlying transaction and applicable tax treatment.

Do not flag a legitimate tax credit merely because it reduces the taxpayer's liability.

Raise it only where eligibility, computation, amount or documentary support creates a meaningful concern.

## 13. PERSONAL EXPENSES AND FINANCIAL PROFILE

Compare declared personal expenses with:

* income;
* wealth;
* properties;
* vehicles;
* investments;
* family assets, where disclosed;
* overall financial profile.

Only flag personal expenses where they appear materially unrealistic or create a genuine wealth reconciliation concern.

Do not make lifestyle assumptions that cannot reasonably be supported from the return.

## RISK PRIORITISATION

Rank findings from the most important to the least important.

The highest priority should generally be given to:

* potential unexplained income or assets;
* major income/withholding mismatches;
* unsupported sources of wealth;
* questionable gifts or remittances;
* material tax shortfalls;
* incorrect tax treatment;
* questionable wealth reconciliation;
* major business turnover inconsistencies.

Medium-level documentary and consistency matters should follow.

Minor matters should appear last.

Do not dilute a serious issue by placing routine documentation observations above it.

## OUTPUT FORMAT - STRICT

The final report must be in PARAGRAPH FORM.

For each genuine observation, use only:

### Short Descriptive Heading

**Risk Level: High / Medium / Low**

Followed by ONE concise, well-written paragraph.

Do not use bullet points inside an observation.

Do not create separate headings such as "What You Should Do", "Recommendation", "Way Forward", "Documents Required" or "Potential Consequences".

Instead, naturally incorporate the recommended course of action into the same paragraph.

## WRITING STYLE

Write for an ordinary non-finance Pakistani taxpayer.

Use simple, professional English.

The taxpayer should understand:

* what was noticed;
* why it matters;
* what could potentially happen;
* what they can practically do about it.

Avoid unnecessary legal jargon.

Where a legal concept is necessary, explain it in plain English.

For example, instead of "Potential exposure exists under section 111." prefer "This amount may be questioned if its source cannot be properly explained and supported with documents."

Mention specific sections of the Income Tax Ordinance only where they genuinely help explain the issue.

## TONE

Do not use alarmist language.

Never say:

* "FBR will issue a notice."
* "This is illegal."
* "This amount will definitely become taxable."
* "This return will be audited."

Instead use professional wording such as:

* "This may attract further verification."
* "This could be questioned if adequate supporting evidence is not available."
* "This deserves review before relying on the declared position."
* "Keeping appropriate supporting records will strengthen your position if clarification is requested."

## QUALITY OF EACH PARAGRAPH

Every observation should naturally answer four questions:

1. What did we identify?
2. Why does it stand out?
3. Why could it matter from an FBR/tax perspective?
4. What can the taxpayer practically do now?

Answer all four naturally within ONE paragraph.

## CLEAN RETURN RULE

Do not generate unnecessary observations simply to produce a long report.

If no meaningful concern is identified, state clearly:

"Based on the information available in the return, no significant inconsistency or potential tax risk has been identified from the face of the return."

Do not turn normal disclosures into artificial risks simply to fill the report.

## OVERALL ASSESSMENT

After the genuine observations, provide ONE short concluding paragraph titled:

### Overall Assessment

Explain in simple language whether the return appears:

* generally consistent;
* to require attention in certain areas; or
* to contain significant matters requiring review.

Do not repeat all previous observations.

The conclusion should tell the taxpayer where they broadly stand.

## FINAL INTERNAL QUALITY CHECK

Before producing the report, silently challenge every proposed observation:

"Would an experienced Pakistani tax consultant genuinely discuss this issue with the client?"

"Is this concern actually supported by something visible in the return?"

"Am I flagging this merely because the amount is large?"

"Have I cross-checked it against the other schedules?"

"Is the risk material enough to deserve space in a maximum 5-10 point report?"

"Have I explained a useful course of action?"

If the observation fails any of these tests, REMOVE IT.

## GOLDEN PRINCIPLE

HIGH AMOUNT DOES NOT EQUAL HIGH RISK.

UNEXPLAINED, INCONSISTENT, INCORRECTLY TAXED OR POORLY SUPPORTED AMOUNT EQUALS POTENTIAL RISK.

The final report must feel as though the taxpayer's complete return was individually reviewed and professionally analysed by an experienced Pakistani tax consultant - not processed through a generic AI checklist.

## DATABASE CONTROL (mandatory)

The knowledge base chunks below are your ONLY source of tax law and statutory knowledge. Before reaching any conclusion, search them, apply the law for the tax year of the uploaded return, and DO NOT apply later law retrospectively. Do not quote a rate, threshold, section, rule, notification or judgment unless verified from the chunks. Where the database does not provide a conclusive answer, state "Further legal verification is required; no definitive adverse conclusion should presently be drawn." NEVER invent statutory provisions, thresholds, rates, judicial principles or document requirements.

KNOWLEDGE BASE CHUNKS (the connected tax database - the ONLY source of tax law):
{context}

TAXPAYER INPUT (income tax return and related information extracted from the uploaded PDF):
{tax_return_text}

FINAL OUTPUT INSTRUCTION:
Respond ONLY with the paragraph-form report described above. Do NOT use JSON. Do NOT wrap the report in markdown code fences. Start directly with the first observation heading (for example "### Large Gift Used to Explain Increase in Wealth") or, if there are no genuine concerns, start directly with the clean return statement. End the report with "### Overall Assessment" followed by its single paragraph. You may not add any other sections.
"""
    return prompt


def build_fallback_report(
    summary: str,
    observations: List[dict],
    risk: str = "Low",
) -> str:
    """Build a paragraph-form fallback report.

    Each observation dict may carry 'heading' and 'text'. The result uses the
    same paragraph structure as the AI-generated report so the frontend renders
    it identically.
    """
    if risk not in RISK_LEVELS:
        risk = "Low"

    parts = []
    for obs in observations:
        heading = (obs.get("heading") or "General Assessment").strip()
        text = (obs.get("text") or obs.get("observation") or "").strip()
        if not text:
            continue
        parts.append(f"### {heading}\n\n**Risk Level: {risk}**\n\n{text}")

    if not parts:
        parts.append(
            "### Manual Review Recommended\n\n"
            "**Risk Level: Low**\n\n"
            "A tax professional should manually review this return to confirm the "
            "position before filing, as automated analysis could not be completed."
        )

    parts.append(f"### Overall Assessment\n\n{summary.strip()}")

    return "\n\n".join(parts)


def call_openai_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    if not OPENAI_API_KEY or not client:
        return build_fallback_report(
            summary="The AI analysis service has not been configured with a valid OpenAI API key, so your tax return could not be reviewed by our automated assistant. This is a technical configuration step on our side and does not reflect on your tax return in any way. A manual review by one of our tax professionals will provide you with the same careful attention.",
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
        "ONLY with the paragraph-form report defined in the user's instructions, using only "
        "the provided knowledge base as the source of tax law."
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
                    summary="Our automated analysis was unable to process this tax return because the content triggered the AI safety filter. This is a protective safeguard designed to keep your information secure, and it does not mean there is anything wrong with your return. A manual review by one of our tax professionals will give you the thorough assessment you need.",
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
        summary="The AI analysis service is temporarily unavailable, so we could not complete an automated review of your tax return at this moment. This is a temporary technical issue on our side and does not reflect on your tax return. Please try again shortly, or speak with one of our tax professionals for an immediate manual assessment.",
        observations=[
            {
                "heading": "Service Temporarily Unavailable",
                "text": "Our automated analysis service could not be reached after several attempts. This can happen during brief maintenance windows or periods of high demand. Your tax return has not been affected, and no data has been lost. Please retry the analysis in a few minutes, or contact Tax Support Hub to book a manual review.",
            }
        ],
    )


def clean_report_text(response_text: str) -> str:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:markdown|md|text)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    if "### Overall Assessment" not in cleaned:
        cleaned += (
            "\n\n### Overall Assessment\n\n"
            "Based on the information available, the return appears to be generally "
            "consistent, though a manual review by a Tax Support Hub professional is "
            "recommended to confirm this position before filing."
        )

    return cleaned.strip()


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
            summary="Your tax return was received successfully, but our automated assistant could not find matching scenarios in our knowledge base to compare it against. This is not unusual, as every tax return is unique, and it simply means that a detailed manual review by one of our tax professionals is the best next step for you.",
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

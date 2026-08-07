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

RISK_RATINGS = ["Critical", "High", "Medium", "Low", "Advisory"]

EVIDENCE_LEVELS = [
    "Level 1 - Strong Evidence",
    "Level 2 - Reasonable Evidence",
    "Level 3 - Weak Evidence",
    "Level 4 - Unsupported",
]

_AREA_OF_CONCERN_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "area",
        "amount",
        "observation",
        "diagnostic",
        "why_it_matters",
        "provision",
        "risk_rating",
        "evidence_level",
        "possible_fbr_query",
        "documents_required",
        "immediate_action",
        "preventive_measure",
    ],
    "properties": {
        "area": {"type": "string"},
        "amount": {"type": "string"},
        "observation": {"type": "string"},
        "diagnostic": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "provision": {"type": "string"},
        "risk_rating": {"type": "string", "enum": RISK_RATINGS},
        "evidence_level": {"type": "string", "enum": EVIDENCE_LEVELS},
        "possible_fbr_query": {"type": "string"},
        "documents_required": {"type": "string"},
        "immediate_action": {"type": "string"},
        "preventive_measure": {"type": "string"},
    },
}

REPORT_JSON_SCHEMA = {
    "name": "tax_risk_review_report",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "overall_risk_rating",
            "overall_summary",
            "risk_counts",
            "total_amount_reviewed",
            "total_amount_exposed",
            "top_five_weaknesses",
            "immediate_actions",
            "filing_recommendation",
            "top_areas_of_concern",
            "risk_matrix",
            "missing_documents_checklist",
            "corrective_action_plan",
            "client_friendly_conclusion",
            "next_step",
        ],
        "properties": {
            "overall_risk_rating": {"type": "string", "enum": RISK_RATINGS},
            "overall_summary": {"type": "string"},
            "risk_counts": {
                "type": "object",
                "additionalProperties": False,
                "required": ["critical", "high", "medium", "low", "advisory"],
                "properties": {
                    "critical": {"type": "integer"},
                    "high": {"type": "integer"},
                    "medium": {"type": "integer"},
                    "low": {"type": "integer"},
                    "advisory": {"type": "integer"},
                },
            },
            "total_amount_reviewed": {"type": "string"},
            "total_amount_exposed": {"type": "string"},
            "top_five_weaknesses": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "immediate_actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 10,
                "items": {"type": "string"},
            },
            "filing_recommendation": {"type": "string"},
            "top_areas_of_concern": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": _AREA_OF_CONCERN_ITEM,
            },
            "risk_matrix": {
                "type": "array",
                "minItems": 1,
                "maxItems": 15,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "ref",
                        "area",
                        "diagnostic_observation",
                        "amount",
                        "potential_exposure",
                        "provision",
                        "risk",
                        "evidence",
                        "corrective_action",
                        "preventive_action",
                    ],
                    "properties": {
                        "ref": {"type": "string"},
                        "area": {"type": "string"},
                        "diagnostic_observation": {"type": "string"},
                        "amount": {"type": "string"},
                        "potential_exposure": {"type": "string"},
                        "provision": {"type": "string"},
                        "risk": {"type": "string", "enum": RISK_RATINGS},
                        "evidence": {"type": "string", "enum": EVIDENCE_LEVELS},
                        "corrective_action": {"type": "string"},
                        "preventive_action": {"type": "string"},
                    },
                },
            },
            "missing_documents_checklist": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {"type": "string"},
            },
            "corrective_action_plan": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "before_filing",
                    "after_filing_before_notice",
                    "on_receipt_of_notice",
                    "future_preventive_controls",
                ],
                "properties": {
                    "before_filing": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 15,
                        "items": {"type": "string"},
                    },
                    "after_filing_before_notice": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 15,
                        "items": {"type": "string"},
                    },
                    "on_receipt_of_notice": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 15,
                        "items": {"type": "string"},
                    },
                    "future_preventive_controls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 15,
                        "items": {"type": "string"},
                    },
                },
            },
            "client_friendly_conclusion": {"type": "string"},
            "next_step": {"type": "string"},
        },
    },
}

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


def build_strict_prompt(tax_return_text: str, chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(chunks) if chunks else "No knowledge base chunks available."

    if len(context) > MAX_CHUNK_CHARS:
        context = context[:MAX_CHUNK_CHARS] + "\n...[truncated]"

    prompt = f"""You are an expert Pakistan Income Tax Return Review, Risk Assessment and Preventive Compliance Assistant for Tax Support Hub.

You have access to a structured and searchable tax knowledge database (below) that may contain: Income Tax Ordinance, 2001; Income Tax Rules, 2002; Finance Act amendments; FBR circulars, notifications, general orders and explanatory notes; relevant judgments and tax principles; return and wealth statement requirements; withholding tax provisions; audit, amendment, penalty and refund provisions; and common factual patterns on which FBR notices and proceedings are initiated.

Your responsibility is NOT merely to identify filing errors. Your PRIMARY responsibility is to identify every material potential risk area, weakness, inconsistency, unsupported position, documentation gap and preventable exposure appearing from the taxpayer's income tax return and related information, so the taxpayer understands: (1) what appears unusual or weak; (2) why it may attract an FBR query; (3) which provision may become relevant; (4) what amount is potentially exposed; (5) what explanation or document is presently missing; (6) what corrective or preventive action to take; and (7) whether the matter should be addressed before filing, through revision, through additional documentation, or only through future record maintenance.

MANDATORY RULES (follow STRICTLY):

1. DATABASE CONTROL: The knowledge base chunks below are your ONLY source of tax knowledge. Before reaching any conclusion, search them, apply the law for the relevant tax year, and DO NOT apply later law retrospectively. Do not quote a rate, threshold, limitation period, section, rule, notification or judgment unless verified from the chunks. Where the database does not provide a conclusive answer, state "Further legal verification is required; no definitive adverse conclusion should presently be drawn." NEVER invent statutory provisions, thresholds, rates, judicial principles or document requirements.

2. MANDATORY SPECIFICITY: Every material observation must identify: what exactly is visible in the return; why it is an area of concern; the amount involved; which provision may apply; what information or evidence is missing; what FBR may potentially ask; what the taxpayer should do now; and what preventive control to follow in future. Generic statements such as "documentation should be maintained" are not acceptable unless the specific transaction, amount, document and corrective action are also identified.

3. ONE-LINE DIAGNOSTIC: For each top area of concern, provide one clear diagnostic sentence in this format: "The return reflects [specific transaction or amount]; however, [specific weakness or inconsistency] is presently visible, which may attract examination under [relevant provision]. The taxpayer should obtain or perform [specific document, reconciliation or corrective action] before [filing, revision, receipt of notice or other relevant stage]." The sentence must be understandable to a non-specialist.

4. RISK SCORING: Assign each issue one of: Critical (material undeclared income, unexplained assets or expenditure, fictitious liability, strong concealment evidence, substantial demand likely); High (material amount, substantially incomplete evidence, strong inconsistency or third-party conflict); Medium (explainable transaction, some evidence, incomplete reconciliation, corrective action can reduce exposure); Low (minor mismatch, procedural weakness, easily correctable); Advisory (no apparent default, preventive improvement recommended). Consider amount, percentage of income or assets, legal strength, documents, third-party data and historical consistency.

5. EVIDENCE QUALITY: Classify the evidence for each material transaction as: "Level 1 - Strong Evidence" (agreement, tax record, banking trail and independent document available); "Level 2 - Reasonable Evidence" (main transaction and payment evidence available, one supporting item missing); "Level 3 - Weak Evidence" (only internal record, self-declaration or incomplete documentation); "Level 4 - Unsupported" (no reliable evidence). Adjust the risk rating based on evidence quality and explain why.

6. PROVISION MAPPING: Consider the following and any other provision supported by the database, applied only for the applicable tax year: sections 4, 11, 12-14 (salary), 15/15A (property), 18-36 (business and deductions), 20-22 (deductions/depreciation), 37/37A (capital gains), 39 (gifts, loans, receipts; other sources), 85/108/109 (relatives, associates, related-party), 111 (unexplained income/assets/expenditure), 114/114A (return, revision, business bank accounts), 116/116A (wealth statement, foreign income/assets), 120 (assessment), 122 (amendment), 149/152/153/161/165 (withholding), 168/170/170A/171 (credits and refunds), 174/176/177/182/205 (records, information, audit, penalties, default surcharge), 236C/236K (property advance tax).

7. KEY ANALYSES: Cover income completeness and classification; wealth statement and reconciliation (opening/closing wealth, additions vs sources, disposal proceeds); the cash-in-hand test (proportion of assets, ratio to income, year-on-year movement, balancing-figure risk); gifts/loans/family transactions (donor identity and capacity, gift deed, transfer trail, donor's tax records); foreign remittances (amount, banking channel, PRC or equivalent certificate, statutory threshold for the year, section 116A); expenses and deductions (business nexus, revenue vs capital, vouchers, payment mode, withholding, related-party arm's length, break-up of miscellaneous expenses); banking-channel and payment-mode compliance; asset additions/disposals (source of funds, agreements, registration data, capital gain); withholding tax and third-party data reconciliation; refund review (source of credit, adjustability, outstanding demands, whether a section 170 application or 170A electronic processing applies - state the applicable limitation from the database); and prior-year/cross-data review (opening balances, historical profit margins, consistent cash).

8. NOTICE EXPOSURE SIMULATION: For every High or Critical item, prepare a likely FBR query in plain language, then give: likely provision; information FBR may request; taxpayer's presently available defence; weakness in that defence; documents required to strengthen it; and whether corrective action should be taken before receiving a notice. NEVER state that FBR will definitely issue a notice.

9. LANGUAGE: Write like an experienced tax consultant explaining to a client, in simple client-friendly language. Use professional, defensible wording. NEVER use "this is illegal", "the tax authority will definitely issue a notice", or "a guaranteed audit". Do not treat every unusual item as an established default - clearly distinguish confirmed non-compliance, probable exposure, potential concern, documentation weakness, reconciliation issue, unusual but explainable transaction, preventive recommendation, and no material concern.

10. QUALITY CONTROL: Do not produce generic advice; every conclusion must be traceable to the taxpayer's data and the knowledge base. Separate legal exposure from documentation exposure. State the taxpayer's possible explanation fairly. Highlight both adverse issues and available tax benefits, including unclaimed tax credits and refund opportunities. State where no material weakness is identified. Do not repeat the same observation under multiple headings. Prioritise material issues over immaterial differences. Assign each recommendation an action priority: Immediate (before filing or within 7 days), Urgent (within 30 days), Important (before any FBR inquiry), Preventive (future years), or No action presently required - and only prescribe a statutory deadline if retrieved from the database.

KNOWLEDGE BASE CHUNKS (the connected tax database - the ONLY source of tax knowledge):
{context}

TAXPAYER INPUT (income tax return and related information extracted from the uploaded PDF):
{tax_return_text}

OUTPUT REQUIREMENTS:
Start with the five most material areas of concern. Do NOT begin with a general explanation of income tax law.
- overall_risk_rating: the overall rating considering the most serious issues found.
- risk_counts: the number of Critical, High, Medium, Low and Advisory observations in the risk_matrix.
- total_amount_reviewed and total_amount_exposed: state amounts in Rupees (for example "Rs. 45,000,000") where data permits, otherwise "Not determinable from the return".
- top_five_weaknesses: the five most serious weaknesses, most serious first.
- immediate_actions: concrete actions required now (before filing or within 7 days).
- filing_recommendation: whether filing should proceed, be held for clarification, or be preceded by revision, stated plainly.
- top_areas_of_concern: EXACTLY 5 objects, ordered highest risk first. For each: area (short name); amount; observation (what appears weak in the return); diagnostic (the one-line diagnostic sentence from rule 3); why_it_matters; provision (section/rule supported by the database, or the "further legal verification" statement); risk_rating; evidence_level; possible_fbr_query (plain-language likely question); documents_required (exact documents); immediate_action; preventive_measure.
- risk_matrix: every material issue (1-15 rows), highest risk first, with ref (1, 2, 3...), area, diagnostic_observation, amount, potential_exposure, provision, risk, evidence, corrective_action, preventive_action.
- missing_documents_checklist: a transaction-specific checklist of exact documents (e.g. gift deed from the identified donor, donor's return and wealth statement, bank statement showing the identified transfer, PRC for the specified remittance, purchase agreement of the identified property, expense break-up for the specified ledger, tax deduction certificate for the identified income, loan agreement and repayment terms, year-wise cash movement, fixed asset and depreciation reconciliation).
- corrective_action_plan: before_filing (complete before submitting), after_filing_before_notice (voluntary documentation, revision or refund action), on_receipt_of_notice (information and legal position to prepare), future_preventive_controls (processes for the next tax year).
- client_friendly_conclusion: plain-language summary of what is satisfactory, what is weak, what requires immediate attention, what can be cured through documentation, what may require revision or tax payment, and what preventive steps to follow.
- next_step: recommend scheduling a free one-to-one review with a Tax Support Hub professional.

If the knowledge base chunks contain NO relevant information for an area, mark it as requiring manual review rather than inventing facts, and state "Further legal verification is required; no definitive adverse conclusion should presently be drawn."
"""
    return prompt


def build_fallback_report(
    summary: str,
    areas: List[dict],
    next_step: str,
    rating: str = "Advisory",
) -> dict:
    generic = {
        "area": "Manual Review Recommended",
        "observation": "A tax professional should manually review this return to confirm the position before filing.",
        "why_it_matters": "Automated analysis could not be completed, so the position cannot be confirmed at this stage.",
        "immediate_action": "Schedule a manual review with a Tax Support Hub professional.",
    }

    def normalize(item: dict) -> dict:
        merged = {**generic, **{k: v for k, v in item.items() if v}}
        return {
            "area": merged.get("area", "General Assessment"),
            "amount": merged.get("amount", "Not determinable from the return"),
            "observation": merged.get("observation", merged.get("explanation", "")),
            "diagnostic": merged.get(
                "diagnostic",
                "A manual review by a Tax Support Hub tax professional is required to confirm this position before filing.",
            ),
            "why_it_matters": merged.get("why_it_matters", merged.get("explanation", "")),
            "provision": merged.get(
                "provision",
                "Further legal verification is required; no definitive adverse conclusion should presently be drawn.",
            ),
            "risk_rating": merged.get("risk_rating", rating),
            "evidence_level": merged.get("evidence_level", "Level 4 - Unsupported"),
            "possible_fbr_query": merged.get(
                "possible_fbr_query", "Not determinable without a manual review."
            ),
            "documents_required": merged.get(
                "documents_required",
                "Complete copy of the income tax return, wealth statement, and all supporting records.",
            ),
            "immediate_action": merged.get(
                "immediate_action", "Schedule a manual review with a Tax Support Hub professional."
            ),
            "preventive_measure": merged.get(
                "preventive_measure",
                "Maintain complete records and supporting documents for future years.",
            ),
        }

    top = [normalize(a) for a in areas]
    while len(top) < 5:
        top.append(normalize(generic))

    risk_matrix = [
        {
            "ref": str(i + 1),
            "area": a["area"],
            "diagnostic_observation": a["observation"],
            "amount": a["amount"],
            "potential_exposure": a["why_it_matters"],
            "provision": a["provision"],
            "risk": a["risk_rating"],
            "evidence": a["evidence_level"],
            "corrective_action": a["immediate_action"],
            "preventive_action": a["preventive_measure"],
        }
        for i, a in enumerate(top)
    ]

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "advisory": 0}
    for a in top:
        key = a["risk_rating"].lower()
        counts[key if key in counts else "advisory"] += 1

    return {
        "overall_risk_rating": rating,
        "overall_summary": summary,
        "risk_counts": counts,
        "total_amount_reviewed": "Not determinable from the return",
        "total_amount_exposed": "Not determinable from the return",
        "top_five_weaknesses": [
            f"{a['area']}: {a['observation']}" for a in top[:5]
        ],
        "immediate_actions": [a["immediate_action"] for a in top],
        "filing_recommendation": "Hold filing pending a manual review by a Tax Support Hub tax professional.",
        "top_areas_of_concern": top,
        "risk_matrix": risk_matrix,
        "missing_documents_checklist": [
            "Complete copy of the income tax return and wealth statement",
            "Wealth reconciliation and tax computation",
            "Withholding tax data and tax deduction certificates",
            "Bank statements for the relevant period",
            "Supporting agreements, invoices and payment evidence",
        ],
        "corrective_action_plan": {
            "before_filing": ["Schedule a manual review before filing the return."],
            "after_filing_before_notice": ["Gather and organise all supporting documentation."],
            "on_receipt_of_notice": [
                "Provide the supporting documents and information requested by the tax authority."
            ],
            "future_preventive_controls": [
                "Maintain complete transaction records, contracts and banking evidence for future years."
            ],
        },
        "client_friendly_conclusion": summary,
        "next_step": next_step,
    }


def call_openai_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    if not OPENAI_API_KEY or not client:
        return json.dumps(build_fallback_report(
            summary="The AI analysis service has not been configured with a valid OpenAI API key, so your tax return could not be reviewed by our automated assistant. This is a technical configuration step on our side and does not reflect on your tax return in any way. A manual review by one of our tax professionals will provide you with the same careful attention.",
            areas=[
                {
                    "area": "Configuration Required",
                    "observation": "The AI analysis service needs a valid OpenAI API key to function, and this key has not yet been configured. Without it, the automated assistant cannot securely read or analyse your tax return. This is entirely a technical setup matter on our side and does not indicate any problem with the document you uploaded.",
                    "immediate_action": "Contact the site administrator so the service can be enabled.",
                }
            ],
            next_step="Please set up your OpenAI API key in the .env file and restart the server, or schedule a free one-to-one review with Tax Support Hub for a manual assessment.",
        ))

    system_prompt = (
        "You are an expert Pakistan Income Tax Return Review, Risk Assessment and Preventive "
        "Compliance Assistant for Tax Support Hub. You identify material risk areas, weaknesses, "
        "inconsistencies, documentation gaps and preventable exposures in tax returns that the "
        "tax authority may question now or in the future. You respond only with valid JSON exactly "
        "matching the structure defined by the response format, using only the provided knowledge base."
    )

    last_error = None
    use_structured_outputs = True
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{max_retries})")
            response_format = (
                {"type": "json_schema", "json_schema": REPORT_JSON_SCHEMA}
                if use_structured_outputs
                else {"type": "json_object"}
            )
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                response_format=response_format,
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
                return json.dumps(build_fallback_report(
                    summary="Our automated analysis was unable to process this tax return because the content triggered the AI safety filter. This is a protective safeguard designed to keep your information secure, and it does not mean there is anything wrong with your return. A manual review by one of our tax professionals will give you the thorough assessment you need.",
                    areas=[
                        {
                            "area": "AI Safety Filter",
                            "observation": "The automated system declined to analyse this document because its content was flagged by the safety filter. These filters are deliberately cautious to protect your privacy and to avoid generating guidance from unclear or restricted material. As a result, we cannot offer automated findings on this occasion.",
                            "why_it_matters": "A qualified tax professional should review this return manually to check its completeness, compliance, deductions, credits and any potential risks.",
                            "immediate_action": "Schedule a manual review with a Tax Support Hub professional.",
                        }
                    ],
                    next_step="Please schedule a free one-to-one review with Tax Support Hub for a thorough manual assessment.",
                ))
            elif use_structured_outputs and ("json_schema" in error_str.lower() or "response_format" in error_str.lower() or "not supported" in error_str.lower() or "400" in error_str or "422" in error_str):
                logger.info("Structured outputs not supported by this endpoint. Falling back to json_object mode.")
                use_structured_outputs = False
                continue
            else:
                if attempt < max_retries - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
                break

    logger.error(f"All OpenAI API retries exhausted. Last error: {last_error}")
    return json.dumps(build_fallback_report(
        summary="The AI analysis service is temporarily unavailable, so we could not complete an automated review of your tax return at this moment. This is a temporary technical issue on our side and does not reflect on your tax return. Please try again shortly, or speak with one of our tax professionals for an immediate manual assessment.",
        areas=[
            {
                "area": "Service Temporarily Unavailable",
                "observation": "Our automated analysis service could not be reached after several attempts. This can happen during brief maintenance windows or periods of high demand. Your tax return has not been affected, and no data has been lost.",
                "immediate_action": "Retry the analysis in a few minutes, or contact Tax Support Hub to book a manual review.",
            }
        ],
        next_step="Please try the analysis again in a few minutes, or schedule a free one-to-one review with Tax Support Hub.",
    ))


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

    if result is None or not isinstance(result, dict):
        logger.warning("Failed to parse AI response as JSON, using fallback")
        return build_fallback_report(
            summary="Our automated assistant returned a response that could not be processed into a readable report. This is a rare technical issue and does not indicate a problem with your tax return. To make sure your return is still reviewed properly, we recommend a manual assessment by one of our tax professionals.",
            areas=[
                {
                    "area": "Analysis Response Issue",
                    "observation": "The AI produced a response that our system could not interpret, so no automated findings could be generated. This occasionally happens when the analysis service returns information in an unexpected format. It does not affect the safety or status of your uploaded document.",
                }
            ],
            next_step="Please schedule a free one-to-one review with Tax Support Hub for a thorough manual assessment.",
        )

    valid_ratings = set(RISK_RATINGS)
    valid_evidence = set(EVIDENCE_LEVELS)

    def s_get(item, key, default=""):
        val = item.get(key)
        return val if isinstance(val, str) else default

    def clean_area(item):
        if not isinstance(item, dict):
            logger.warning(f"Skipping malformed area entry: {type(item).__name__}")
            return None
        rating = s_get(item, "risk_rating")
        if rating not in valid_ratings:
            rating = "Advisory"
        evidence = s_get(item, "evidence_level")
        if evidence not in valid_evidence:
            evidence = "Level 3 - Weak Evidence"
        explanation = s_get(item, "explanation")
        return {
            "area": s_get(item, "area") or "General Assessment",
            "amount": s_get(item, "amount") or "Not determinable from the return",
            "observation": s_get(item, "observation") or explanation,
            "diagnostic": s_get(item, "diagnostic")
            or "A manual review by a tax professional is required to confirm this position before filing.",
            "why_it_matters": s_get(item, "why_it_matters") or explanation,
            "provision": s_get(item, "provision")
            or "Further legal verification is required; no definitive adverse conclusion should presently be drawn.",
            "risk_rating": rating,
            "evidence_level": evidence,
            "possible_fbr_query": s_get(item, "possible_fbr_query")
            or "Not determinable without further review.",
            "documents_required": s_get(item, "documents_required")
            or "Supporting documents for the transaction described.",
            "immediate_action": s_get(item, "immediate_action")
            or "Obtain the supporting documents and consult a tax professional.",
            "preventive_measure": s_get(item, "preventive_measure")
            or "Maintain complete records for future years.",
        }

    def clean_matrix(item):
        if not isinstance(item, dict):
            logger.warning(f"Skipping malformed matrix entry: {type(item).__name__}")
            return None
        rating = s_get(item, "risk")
        if rating not in valid_ratings:
            rating = "Advisory"
        evidence = s_get(item, "evidence")
        if evidence not in valid_evidence:
            evidence = "Level 3 - Weak Evidence"
        return {
            "ref": s_get(item, "ref"),
            "area": s_get(item, "area") or "General",
            "diagnostic_observation": s_get(item, "diagnostic_observation")
            or s_get(item, "observation"),
            "amount": s_get(item, "amount") or "Not determinable",
            "potential_exposure": s_get(item, "potential_exposure")
            or s_get(item, "why_it_matters"),
            "provision": s_get(item, "provision")
            or "Further legal verification is required.",
            "risk": rating,
            "evidence": evidence,
            "corrective_action": s_get(item, "corrective_action")
            or s_get(item, "immediate_action"),
            "preventive_action": s_get(item, "preventive_action")
            or s_get(item, "preventive_measure"),
        }

    def clean_str_list(value, default):
        if not isinstance(value, list):
            return list(default)
        out = [v for v in value if isinstance(v, str) and v.strip()]
        return out or list(default)

    rating = result.get("overall_risk_rating")
    result["overall_risk_rating"] = rating if rating in valid_ratings else "Advisory"
    result["overall_summary"] = s_get(result, "overall_summary") or "Your tax return has been reviewed."
    result["total_amount_reviewed"] = (
        result.get("total_amount_reviewed")
        if isinstance(result.get("total_amount_reviewed"), str)
        else "Not determinable from the return"
    )
    result["total_amount_exposed"] = (
        result.get("total_amount_exposed")
        if isinstance(result.get("total_amount_exposed"), str)
        else "Not determinable from the return"
    )

    areas_raw = result.get("top_areas_of_concern")
    if not isinstance(areas_raw, list):
        areas_raw = []
    areas = [a for a in (clean_area(a) for a in areas_raw) if a is not None]
    result["top_areas_of_concern"] = areas

    matrix_raw = result.get("risk_matrix")
    if not isinstance(matrix_raw, list):
        matrix_raw = []
    matrix = []
    for idx, item in enumerate(matrix_raw):
        entry = clean_matrix(item)
        if entry:
            if not entry["ref"]:
                entry["ref"] = str(idx + 1)
            matrix.append(entry)
    result["risk_matrix"] = matrix

    result["top_five_weaknesses"] = clean_str_list(
        result.get("top_five_weaknesses"), ["See the top areas of concern below."]
    )[:5]
    result["immediate_actions"] = clean_str_list(
        result.get("immediate_actions"), ["Consult a tax professional."]
    )
    result["missing_documents_checklist"] = clean_str_list(
        result.get("missing_documents_checklist"),
        ["Organise supporting documentation for the matters identified."],
    )

    plan = result.get("corrective_action_plan")
    if not isinstance(plan, dict):
        plan = {}
    for key in (
        "before_filing",
        "after_filing_before_notice",
        "on_receipt_of_notice",
        "future_preventive_controls",
    ):
        plan[key] = clean_str_list(plan.get(key), [])
    result["corrective_action_plan"] = plan

    result["filing_recommendation"] = s_get(result, "filing_recommendation") or (
        "Hold filing pending a manual review by a tax professional."
    )
    result["client_friendly_conclusion"] = s_get(
        result, "client_friendly_conclusion"
    ) or s_get(result, "overall_summary")
    result["next_step"] = s_get(result, "next_step") or (
        "Schedule a free one-to-one review with Tax Support Hub."
    )

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "advisory": 0}
    for entry in matrix:
        key = entry["risk"].lower()
        counts[key if key in counts else "advisory"] += 1
    if sum(counts.values()) == 0:
        for area in areas:
            key = area["risk_rating"].lower()
            counts[key if key in counts else "advisory"] += 1
    result["risk_counts"] = counts

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
        return build_fallback_report(
            summary="Your tax return was received successfully, but our automated assistant could not find matching scenarios in our knowledge base to compare it against. This is not unusual, as every tax return is unique, and it simply means that a detailed manual review by one of our tax professionals is the best next step for you.",
            areas=[
                {
                    "area": "Knowledge Base Coverage",
                    "observation": "Our automated system could not find tax scenarios in our knowledge base that closely match the details in your specific return. This happens when a return is unique or contains uncommon arrangements, and it is not a cause for concern. It simply means the automated assistant could not draw a reliable comparison, so a human expert is better placed to give you accurate guidance.",
                },
                {
                    "area": "Completeness Check",
                    "observation": "We were unable to automatically verify that all required sections of your tax return are complete, including your personal details, income statements, deductions and any supporting schedules. A tax professional will review every section of your return to make sure nothing has been overlooked.",
                },
                {
                    "area": "Compliance Review",
                    "observation": "Automated compliance checking could not be completed because there was insufficient reference material to compare against your return, including the correct treatment of income, expenses and reporting requirements. A qualified professional will check your return against the current rules.",
                },
                {
                    "area": "Deduction & Credit Analysis",
                    "observation": "We could not automatically verify whether all the deductions and tax credits you may be entitled to have been claimed on your return. A professional review of your expenses and entitlements is recommended so that every benefit you are owed is claimed.",
                },
                {
                    "area": "Risk Assessment",
                    "observation": "A full risk assessment involves checking for errors, inconsistencies or anything that could attract attention from the tax authority. This requires judgement and experience, which our automated assistant is not able to provide without a reliable knowledge base match.",
                },
            ],
            next_step="Schedule a free one-to-one review with a Tax Support Hub professional to get a thorough analysis of your tax return.",
        )

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

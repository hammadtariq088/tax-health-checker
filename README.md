# Tax Health Checker

An AI-powered tax return health check tool built for **Tax Support Hub** (https://taxsupporthub.com/).

Users upload a PDF of their tax return and receive a **Pakistan Individual Income Tax Return Risk Review** — a paragraph-form report of genuine potential areas of concern (each with a descriptive heading, a High/Medium/Low risk level, and a plain-language explanation with practical next steps), followed by an Overall Assessment — plus PDF download and WhatsApp booking integration.

---

## How It Works

1. Upload a tax return PDF
2. Text is extracted from the PDF using pdfplumber
3. Knowledge chunks are embedded using **OpenAI** (`text-embedding-3-small`) via a one-time script (`embed_kb.py`)
4. Relevant chunks are retrieved from **pgvector** (PostgreSQL) using cosine similarity
5. OpenAI (`gpt-4o`) performs a face-of-return tax risk review using the master prompt (cross-checking the return of income, wealth statement, reconciliation, withholding, and all schedules together) against your knowledge base only (`temperature=0.0`)
6. A professional paragraph-form report is returned — always a full area-by-area review. Each applicable area is its own section with a heading and a High/Medium/Low risk level: genuine concerns are flagged with their real risk level, and clean areas are reported as "confirmed consistent" (Risk Level: Low). The report always ends with an Overall Assessment, so output is consistent regardless of how clean the return is
7. The report can be downloaded as PDF (email/phone collected first) or shared via WhatsApp

**No user data is stored — uploaded PDFs are processed in memory only.**

---

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key — for AI analysis and embeddings
- A PostgreSQL database with **pgvector** extension (use [Neon](https://neon.tech/) or [Supabase](https://supabase.com/))

---

## Setup Instructions

### 1. Set Up PostgreSQL with pgvector

**Option A — Neon (serverless, recommended):**

1. Go to https://neon.tech/ and sign up
2. Create a project
3. Copy the connection string from the dashboard

**Option B — Supabase:**

1. Go to https://supabase.com/ and sign up
2. Create a new project
3. Go to Project Settings → Database → Connection string
4. Copy the URI connection string

**Option C — Local PostgreSQL:**

```bash
sudo apt install postgresql postgresql-contrib postgresql-16-pgvector
createdb tax_health_checker
```

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd tax-health-checker
```

### 3. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Get an API Key

**OpenAI API Key** (for AI analysis and embeddings):

1. Go to https://platform.openai.com/api-keys
2. Click **"Create new secret key"**
3. Copy the key (starts with `sk-`)
4. Make sure you have credits/usage access on your OpenAI account

### 6. Configure Environment

Edit the `.env` file in the project root:

```
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

> Replace `DATABASE_URL` with your actual Neon, Supabase, or local PostgreSQL connection string.
> The `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` values are optional and fall back to the defaults shown above.

### 7. Add Knowledge Base Files

Place your exported ChatGPT conversation files (.txt or .json) in the `knowledge_base/` folder:

```bash
knowledge_base/
├── conversations-000.txt
├── chat_about_tax_deductions.json
└── ...
```

### 8. Embed the Knowledge Base (Required — one-time)

```bash
python3 embed_kb.py
```

This reads all files from `knowledge_base/`, chunks them (8000 chars each), generates embeddings via the OpenAI API, and stores them in pgvector. Run this once after adding/updating files.

### 9. Run the Server

```bash
python3 main.py
```

The app will be available at **http://localhost:8000**

To re-embed after updating knowledge base files, run `python3 embed_kb.py --no-clear` (skips truncation) or `python3 embed_kb.py` (truncates and re-embeds everything).

---

## API Endpoints

| Endpoint                     | Method | Description                                           |
| ---------------------------- | ------ | ----------------------------------------------------- |
| `/`                          | GET    | Frontend HTML page                                    |
| `/api/health`                | GET    | Health check + knowledge base status                  |
| `/api/kb-status`             | GET    | Knowledge base chunk count                            |
| `/api/health-check`          | POST   | Upload a PDF and get a health report                  |
| `/api/capture-lead`          | POST   | Store email/phone for follow-up (before PDF download) |
| `/api/reload-knowledge-base` | POST   | Clear knowledge base (re-run embed_kb.py to reload)   |

### Example: Health Check via API

```bash
curl -X POST http://localhost:8000/api/health-check \
  -F "file=@/path/to/tax_return.pdf"
```

### Example: Check Knowledge Base Status

```bash
curl http://localhost:8000/api/kb-status
# {"ready": true, "chunks": 362, "message": "Knowledge base: 362 chunks"}
```

---

## Important Notes

- **Embeddings** are generated via the OpenAI API (`text-embedding-3-small`, 1536 dims) — no local ML models, no heavy CPU/RAM usage
- **AI analysis** uses OpenAI (`gpt-4o` by default, configurable via `OPENAI_MODEL`) at `temperature=0.0` and returns a plain-text paragraph-form report (no JSON schema)
- **Vector search** uses pgvector with cosine similarity (`<=>` operator)
- **Knowledge base embedding** is a separate one-time step (`embed_kb.py`) — the server starts instantly without waiting
- **No hardcoded tax rules** — the AI only uses your ChatGPT exports as its knowledge source, and must never invent provisions or thresholds; it flags "Further legal verification is required" when the knowledge base is inconclusive
- **Risk levels** — each observation is graded High / Medium / Low only
- **Lead capture** — when downloading the PDF report, email and phone are stored in the `leads` table for follow-up
- **No user data stored from PDFs** — uploaded PDFs are processed in memory and discarded
- **PDF download** — the report can be downloaded as a PDF using html2pdf.js (client-side generation)
- **WhatsApp booking** — floating action button + inline button link to WhatsApp deep-link for booking reviews
- **Animated UI** — scanning animation with progress steps, staggered area card animations
- **Rate limit handling** — retry logic with exponential backoff for both embedding and generation APIs
- **If no relevant knowledge is found**, the system returns a manual-review response
- **PDF requirement** — works with text-based PDFs only. Use OCR for scanned documents first.

---

## File Structure

```
tax-health-checker/
├── .env                 # Environment variables (OPENAI_API_KEY, DATABASE_URL)
├── .gitignore
├── requirements.txt     # Python dependencies
├── main.py              # FastAPI application (PDF extraction, RAG, OpenAI analysis, lead capture)
├── embed_kb.py          # One-time knowledge base embedding script
├── README.md            # This file
├── knowledge_base/      # Put Converted Conversations.txt files here
├── chatgpt-export/      # Put Conversations.json files here
├── templates/
│   └── index.html       # Frontend HTML (Tailwind CSS, html2pdf.js, WhatsApp integration)
```

---

## How to Get ChatGPT Exports

1. Go to https://chat.openai.com/
2. Click on your profile → **Settings** → **Data Controls**
3. Click **"Export Data"** → **"Confirm Export"**
4. You'll receive an email with a download link
5. Extract the zip file and place relevant .json or .txt files in `knowledge_base/`

---

## Deploy to Production

Set environment variables on your hosting platform:

- `OPENAI_API_KEY` — OpenAI API key for AI analysis and embeddings
- `DATABASE_URL` — PostgreSQL connection string (use a cloud provider)

For scale-to-zero platforms (Koyeb, Railway, etc.), ensure `embed_kb.py` is run as a one-off build step or deploy hook before the server starts.

---

## License

Internal tool for Tax Support Hub.

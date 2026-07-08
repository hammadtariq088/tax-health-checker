# Tax Health Checker

An AI-powered tax return health check tool built for **Tax Support Hub** (https://taxsupporthub.com/).

Users upload a PDF of their tax return and receive a simple, non-technical health report with a status rating, key areas of focus, and a call-to-action for a free professional review.

---

## How It Works

1. Upload a tax return PDF
2. Text is extracted from the PDF using pdfplumber
3. Knowledge chunks are embedded using **Gemini text-embedding-004** via API
4. Relevant chunks are retrieved from **pgvector** (PostgreSQL) using cosine similarity
5. Google Gemini AI analyzes the return using only your knowledge base
6. A clean report is returned with overall health status, 5 key areas, and next steps

**No user data is stored — uploaded PDFs are processed in memory only.**

---

## Prerequisites

- Python 3.10 or higher
- A Google Gemini API key (free tier available)
- A PostgreSQL database with **pgvector** extension (use [Supabase free tier](https://supabase.com/) or [Neon](https://neon.tech/))

---

## Setup Instructions

### 1. Set Up PostgreSQL with pgvector

**Option A — Supabase (recommended, free):**
1. Go to https://supabase.com/ and sign up
2. Create a new project
3. Go to Project Settings → Database → Connection string
4. Copy the URI connection string

**Option B — Neon (serverless PostgreSQL):**
1. Go to https://neon.tech/ and sign up
2. Create a project
3. Copy the connection string from the dashboard

**Option C — Local PostgreSQL:**
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib
# Install pgvector
sudo apt install postgresql-16-pgvector  # adjust version to match your PostgreSQL
# Create database
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

### 5. Get a Google Gemini API Key

1. Go to https://aistudio.google.com/app/apikey
2. Click **"Create API Key"**
3. Copy the key
4. Edit the `.env` file in the project root:

```
GEMINI_API_KEY=your_actual_api_key_here
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

> Replace `DATABASE_URL` with your actual Supabase, Neon, or local PostgreSQL connection string.

### 6. Add Knowledge Base Files

Place your exported ChatGPT conversation files (.txt or .json) in the `knowledge_base/` folder:

```bash
knowledge_base/
├── chat_about_tax_deductions.txt
├── irs_audit_discussion.json
└── ...
```

The tool will automatically process these files on startup. To reload the knowledge base after adding new files, call the `/api/reload-knowledge-base` endpoint (see below).

### 7. Run the Tool Locally

```bash
python3 main.py
```

The app will be available at **http://localhost:8000**

---

## API Endpoints

| Endpoint                     | Method | Description                          |
| ---------------------------- | ------ | ------------------------------------ |
| `/`                          | GET    | Frontend HTML page                   |
| `/api/health`                | GET    | Health check + knowledge base status |
| `/api/health-check`          | POST   | Upload a PDF and get a health report |
| `/api/reload-knowledge-base` | POST   | Reload knowledge base from files     |

### Example: Health Check via API

```bash
curl -X POST http://localhost:8000/api/health-check \
  -F "file=@/path/to/tax_return.pdf"
```

### Example: Reload Knowledge Base

```bash
curl -X POST http://localhost:8000/api/reload-knowledge-base
```

---

## Important Notes

- **Embeddings** are generated via Gemini API (`text-embedding-004`) — no local ML models, no heavy CPU/RAM usage
- **Vector search** uses pgvector with cosine similarity (`<=>` operator)
- **No hardcoded tax rules** — the AI only uses your ChatGPT exports as its knowledge source
- **No user data stored** — uploaded PDFs are processed in memory and discarded
- **Rate limit handling** — automatic retry with exponential backoff for both embedding and generation APIs
- **If no relevant knowledge is found**, the system returns a manual-review response
- **PDF requirement** — works with text-based PDFs only. Use OCR for scanned documents first.

---

## Deploy to Koyeb (Free)

1. Push your code to GitHub:

```bash
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/tax-health-checker.git
git push -u origin main
```

2. Go to [Koyeb.com](https://www.koyeb.com/) and create an account
3. Click **"Create App"** → **"Docker"** → select your GitHub repository
4. In **"Instance"** section, choose the free plan
5. Set the **Run Command** to: `python main.py`
6. Add environment variables:
   - `GEMINI_API_KEY`
   - `DATABASE_URL` (use a cloud PostgreSQL like Supabase/Neon)
7. Click **"Deploy"**

Koyeb will automatically build and deploy your app. You'll get a public URL like `https://tax-health-checker-xxxx.koyeb.app`.

---

## Deploy to Hostinger VPS (Alternative)

1. SSH into your VPS
2. Install Python 3.10+ and pip
3. Clone the repository
4. Set up the virtual environment and install dependencies
5. Set up a systemd service or use supervisor to keep the app running
6. Optionally set up Nginx as a reverse proxy

```bash
# Example: Run with nohup
nohup python main.py > app.log 2>&1 &
```

---

## Embed into WordPress (iframe)

Add this code to any WordPress page or post (using the "Custom HTML" block or a plugin like "Insert Headers and Footers"):

```html
<iframe
  src="https://your-deployed-url.koyeb.app/"
  width="100%"
  height="800"
  frameborder="0"
  style="border: none; max-width: 100%; overflow: hidden;"
  allow="clipboard-read; clipboard-write"
></iframe>
```

Replace `https://your-deployed-url.koyeb.app/` with your actual deployment URL.

---

## Managing Knowledge Base via WordPress

The client can update the knowledge base using a WordPress file manager plugin:

1. Install a WordPress file manager plugin (e.g., "File Manager" by mndpsingh287)
2. Navigate to the file manager in the WordPress admin dashboard
3. Create or navigate to a folder like `/wp-content/knowledge_base/`
4. Upload new ChatGPT export files (.txt or .json) to this folder
5. Use a WordPress custom endpoint or a simple PHP page to call the reload API:

```php
<?php
// Place this in a custom WordPress plugin or theme file
$response = wp_remote_post('https://your-deployed-url.koyeb.app/api/reload-knowledge-base');
if (!is_wp_error($response)) {
    echo 'Knowledge base reloaded successfully.';
}
?>
```

Or simply use a browser to visit:

```
https://your-deployed-url.koyeb.app/api/reload-knowledge-base
```

---

## File Structure

```
tax-health-checker/
├── .env                 # Environment variables (GEMINI_API_KEY)
├── .gitignore
├── requirements.txt     # Python dependencies
├── main.py              # FastAPI application
├── README.md            # This file
├── knowledge_base/      # Put ChatGPT exports here
├── templates/
│   └── index.html       # Frontend HTML
```

---

## How to Get ChatGPT Exports

1. Go to https://chat.openai.com/
2. Click on your profile → **Settings** → **Data Controls**
3. Click **"Export Data"** → **"Confirm Export"**
4. You'll receive an email with a download link
5. Extract the zip file and place relevant .json or .txt files in `knowledge_base/`

---

---

## License

Internal tool for Tax Support Hub.

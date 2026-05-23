# Company Research Agent

An autonomous company research and due-diligence pipeline built with LangGraph, OpenAI, Pinecone, Cohere, FastAPI, and n8n.

It takes a single company name, researches six investment topics, runs retrieval and reranking, generates a structured report, and saves a polished PDF plus backup outputs.

## What It Does

- Validates the input company name
- Runs live web research across six due-diligence topics
- Chunks and embeds research into Pinecone
- Retrieves and reranks the most relevant context
- Generates a six-section investment report
- Saves a formatted PDF report locally
- Optionally sends the report to Telegram and stores a shareable link

## Output Report

The generated report uses this structure:

1. Executive Summary
2. Business Model
3. Market Analysis
4. Competitive Landscape
5. Risk Factors
6. Investment Recommendation

The PDF template includes:

- a branded cover page
- metric cards for ticket, risk, opportunity, and date
- best-effort company logo lookup
- color-coded section headers
- cleaner spacing and page styling

## Project Layout

- `agent_server.py` - FastAPI app and LangGraph workflow
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment config
- `docs/` - setup and deployment notes
- `outputs/` - generated JSON artifacts and saved runs
- `sprint1_foundation.ipynb` - initial state and validation work
- `sprint2_research.ipynb` - research and chunking
- `sprint3_rag_agent.ipynb` - retrieval, reranking, analysis
- `sprint4_5_report_pipeline.ipynb` - report generation and end-to-end flow
- `user_stories.md` - project notes and scope
- `AGENTS.md` - internal workflow reference for the agent

## Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...
COHERE_API_KEY=...
NOTION_TOKEN=secret_...
NOTION_DB_ID=...
PUBLIC_BASE_URL=https://your-public-host
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

`NOTION_TOKEN`, `NOTION_DB_ID`, and Telegram variables are optional.

If you are using the live Render deployment, set:

```bash
PUBLIC_BASE_URL=https://research-agent-2-sl8r.onrender.com
```

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:

```bash
python agent_server.py
```

4. Confirm the server is healthy:

```text
http://127.0.0.1:8000/health
```

## API Endpoints

- `GET /health` - health check and runtime info
- `GET /config` - current server configuration
- `POST /research` - run the full company research pipeline
- `GET /reports/{filename}` - download a generated PDF report

## n8n Integration

Use the `POST /research` endpoint in your HTTP Request node.

Example payload:

```json
{
  "company_name": "Anthropic"
}
```

If you are using a public tunnel or deployment, point n8n to the public `/research` URL for the running server.

For your Render deployment, that endpoint is:

```text
https://research-agent-2-sl8r.onrender.com/research
```

## Render Deployment

The project is configured for Render as a Python web service. 

- Build command: `pip install -r requirements.txt`
- Start command: `python agent_server.py`
- The app reads `PORT` from the environment

## Generated Files

The server writes report PDFs to:

```text
.generated_reports/
```

Generated notebooks, JSON outputs, and PDFs can stay in the workspace for review, but they are treated as outputs rather than source code.

## Notes

- The repo keeps `AGENTS.md` as the internal instruction file for the autonomous agent.
- `README.md` is the clean public entry point for the project.
- Company logos are fetched best-effort from the discovered research domain, so some reports may show a logo and some may show a placeholder snapshot instead.

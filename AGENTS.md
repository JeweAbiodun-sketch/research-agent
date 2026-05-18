# AGENTS.md
## Autonomous Company Research & Report Generation Agent

**Project:** Module 3 — Individual Project  
**Framework:** LangGraph + OpenAI + Pinecone + Cohere  
**Version:** 1.0  

---

## Agent Name & Purpose

**Name:** CompanyResearchAgent  
**Purpose:** Autonomously research any company and generate a structured 6-section investment due diligence report. Takes a single company name as input and delivers a complete report with no manual steps.

---

## Tools Available

| Tool | Node | When to Use | API |
|------|------|-------------|-----|
| OpenAI Web Search | `research_node` | Always — for all 6 research topics | `openai.responses.create` with `web_search_preview` |
| OpenAI Embeddings | `embed_and_store_node` | After research — to vectorise chunks | `openai.embeddings.create` (text-embedding-3-small) |
| Pinecone Upsert | `embed_and_store_node` | After embedding — to store vectors | `pinecone_index.upsert` |
| Pinecone Query | `retrieve_node` | After research is stored — to retrieve | `pinecone_index.query` |
| Cohere Rerank | `rerank_node` | After Pinecone retrieval — to filter | `cohere_client.rerank` |
| OpenAI Chat | `analyse_node` | After reranking — to classify | `openai.chat.completions.create` |
| OpenAI Chat | `generate_report_node` | After analysis — to write sections | `openai.chat.completions.create` (gpt-4o) |
| Notion API | `save_report_node` | After report generation — to save | `POST /v1/pages` |

---

## Research Workflow (Step-by-Step)

```
1. VALIDATE   → Clean company name, generate ticket ID (RES-YYYYMMDD-XXXXXXXX)
2. RESEARCH   → Query OpenAI web search on 6 due diligence topics
3. EMBED      → Chunk text (500 words, 50 overlap) → embed → upsert to Pinecone
4. RETRIEVE   → Embed query → Pinecone top-10 similarity search
5. RERANK     → Cohere cross-encoder → top-3 most relevant chunks
6. ANALYSE    → LangGraph agent classifies risk, opportunity, priority
7. REPORT     → GPT-4o generates 6 sections using reranked context
8. SAVE       → Notion page created + local Markdown backup
```

---

## Output Format — 6-Section Report

| # | Section | Content |
|---|---------|---------|
| 1 | Executive Summary | Overview, stage, sector, headline metrics |
| 2 | Business Model | Revenue streams, pricing, customer segments |
| 3 | Market Analysis | TAM/SAM/SOM, growth rate, regulatory environment |
| 4 | Competitive Landscape | Top 3–5 competitors, moat assessment |
| 5 | Risk Factors | Tech/market/team/regulatory/financial risks (Low/Medium/High) |
| 6 | Investment Recommendation | Score (1–10), Priority (Pass/Watch/Invest), next steps |

---

## Error Behaviour

| Error Type | Behaviour |
|-----------|-----------|
| Empty company name | Pipeline stops at `validate_input` — returns error message |
| OpenAI API failure | 3 retries with exponential backoff (1s, 2s, 4s) |
| Pinecone upsert failure | 3 retries — if all fail, error logged and pipeline stops |
| Cohere rerank failure | Falls back to top-3 Pinecone similarity results |
| Report section failure | Section marked `[Generation failed]` — other sections continue |
| Notion save failure | Falls back to local Markdown file |
| Any node failure | `error_handler_node` fires — logs all errors, alerts analyst |

---

## Skills Reference

| Skill File | Covers |
|-----------|--------|
| `skills/research.md` | Web search prompts, topic coverage, chunking strategy |
| `skills/classification.md` | Risk scoring rubric, opportunity scoring, JSON schema |
| `skills/report_gen.md` | Section prompts, word limits, citation format |

---

## Environment Variables Required

```bash
OPENAI_API_KEY    = sk-...
PINECONE_API_KEY  = pcsk_...
COHERE_API_KEY    = ...
NOTION_TOKEN      = secret_...   # Optional
NOTION_DB_ID      = ...          # Optional
```

---

## Notebook Structure

| Notebook | Sprint | Covers |
|---------|--------|--------|
| `sprint1_foundation.ipynb` | Day 1 | AgentState schema, validation, acknowledgement |
| `sprint2_research.ipynb` | Day 2 | OpenAI web search, chunking, Pinecone storage |
| `sprint3_rag_agent.ipynb` | Day 3 | Pinecone retrieval, Cohere reranking, LangGraph analysis |
| `sprint4_5_report_pipeline.ipynb` | Day 4–5 | Report generation, full pipeline, live demo |

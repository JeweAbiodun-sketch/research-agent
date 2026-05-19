"""
agent_server.py
FastAPI server that wraps the LangGraph research pipeline..
n8n calls this server via HTTP POST /research.

Run with:
    python agent_server.py

Then test with:
    curl -X POST http://localhost:8000/research \
      -H "Content-Type: application/json" \
      -d '{"company_name": "Anthropic"}'
"""

import os, json, time, uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY",   "")
NOTION_TOKEN     = os.getenv("NOTION_TOKEN",     "")
NOTION_DB_ID     = os.getenv("NOTION_DB_ID",     "")

assert OPENAI_API_KEY,   "❌ OPENAI_API_KEY not set in .env"
assert PINECONE_API_KEY, "❌ PINECONE_API_KEY not set in .env"
assert COHERE_API_KEY,   "❌ COHERE_API_KEY not set in .env"

# ── Import AI clients ─────────────────────────────────────────────────────────
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import cohere
from langgraph.graph import StateGraph, END
from typing import TypedDict

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc            = Pinecone(api_key=PINECONE_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY)

PINECONE_INDEX_NAME = "company-research-agent"
EMBED_MODEL         = "text-embedding-3-small"
RESEARCH_MODEL      = "gpt-4o-mini"
REPORT_MODEL        = "gpt-4o"

# ── Pinecone index ────────────────────────────────────────────────────────────
def get_or_create_index():
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
        print("✅ Pinecone index created")
    return pc.Index(PINECONE_INDEX_NAME)

pinecone_index = get_or_create_index()
print("✅ Pinecone index ready")

# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    company_name:      str
    ticket_id:         str
    initiated_at:      str
    status:            str
    errors:            List[str]
    raw_research:      Optional[str]
    research_chunks:   Optional[List[str]]
    pinecone_ids:      Optional[List[str]]
    retrieved_chunks:  Optional[List[str]]
    reranked_chunks:   Optional[List[str]]
    risk_score:        Optional[str]
    opportunity_score: Optional[int]
    retry_count:       int
    report_sections:   Optional[Dict[str, str]]
    notion_url:        Optional[str]
    report_ready:      bool
    workflow_path:     List[str]


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE NODES
# ══════════════════════════════════════════════════════════════════════════════

RESEARCH_TOPICS = [
    "company overview, founding year, headquarters, mission, and key products or services",
    "funding history, total raised, investors, valuation, and latest funding round",
    "founding team, CEO background, key executives, and leadership",
    "recent news, product launches, and notable events in the last 12 months",
    "main competitors, market positioning, and competitive advantages",
    "market size, growth trends, and regulatory environment",
]

REPORT_SECTIONS = [
    ("executive_summary",         "Executive Summary"),
    ("business_model",            "Business Model"),
    ("market_analysis",           "Market Analysis"),
    ("competitive_landscape",     "Competitive Landscape"),
    ("risk_factors",              "Risk Factors"),
    ("investment_recommendation", "Investment Recommendation"),
]

SECTION_PROMPTS = {
    "executive_summary":         "Write a 150-word executive summary for {company}. Cover: what they do, founding year, stage, sector, and headline metrics.",
    "business_model":            "Describe {company}'s business model in 150 words. Cover: revenue streams, customer segments, pricing, and unit economics.",
    "market_analysis":           "Analyse the market for {company} in 150 words. Cover: TAM/SAM/SOM, growth rate, key trends, regulatory environment.",
    "competitive_landscape":     "Map the competitive landscape for {company} in 150 words. Name 3-5 competitors, differentiation, and moat strength.",
    "risk_factors":              "List key risk factors for investing in {company} in 150 words. Rate each Low/Medium/High. Cover tech, market, team, regulatory, financial risks.",
    "investment_recommendation": "Write an investment recommendation for {company} in 150 words. Include overall score (1-10), rating (Pass/Watch/Invest), top 3 strengths, top 3 concerns, next steps.",
}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    step  = chunk_size - overlap
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), step) if words[i:i+chunk_size]]


# ── Node 1: Validate ──────────────────────────────────────────────────────────
def validate_input_node(state: AgentState) -> AgentState:
    raw    = state.get("company_name", "").strip()
    errors = list(state.get("errors", []))
    print(f"[VALIDATE] '{raw}'")

    if not raw or len(raw) < 2:
        return {**state, "status": "error_invalid_input",
                "errors": errors + ["Invalid company name"],
                "workflow_path": state.get("workflow_path", []) + ["validate"]}

    ticket = f"RES-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    print(f"[VALIDATE] ✅ Ticket: {ticket}")
    return {**state,
            "company_name":  " ".join(raw.split()),
            "ticket_id":     ticket,
            "initiated_at":  datetime.now(timezone.utc).isoformat(),
            "status":        "validated",
            "errors":        errors,
            "workflow_path": state.get("workflow_path", []) + ["validate"]}


# ── Node 2: Research ──────────────────────────────────────────────────────────
def research_node(state: AgentState) -> AgentState:
    company = state["company_name"]
    errors  = list(state.get("errors", []))
    results = []
    print(f"[RESEARCH] Researching '{company}' across {len(RESEARCH_TOPICS)} topics...")

    for i, topic in enumerate(RESEARCH_TOPICS, 1):
        prompt = f"Research this about {company}: {topic}. Provide factual, specific information with numbers and dates."
        print(f"  [{i}/{len(RESEARCH_TOPICS)}] {topic[:55]}...")

        for attempt in range(3):
            try:
                resp = openai_client.responses.create(
                    model=RESEARCH_MODEL,
                    tools=[{"type": "web_search_preview"}],
                    input=prompt
                )
                text = ""
                for block in resp.output:
                    if hasattr(block, "content"):
                        for c in block.content:
                            if hasattr(c, "text"):
                                text += c.text
                if text.strip():
                    results.append(f"### {topic.upper()}\n{text.strip()}")
                    break
            except Exception as e:
                print(f"    ⚠️ Attempt {attempt+1}/3: {e}")
                if attempt == 2:
                    errors.append(f"Research topic failed: {e}")
                else:
                    time.sleep(2 ** attempt)

    if not results:
        return {**state, "status": "error_research_failed",
                "errors": errors + ["All research topics failed"],
                "workflow_path": state.get("workflow_path", []) + ["research"]}

    raw = f"# RESEARCH: {company}\n\n" + "\n\n".join(results)
    print(f"[RESEARCH] ✅ {len(raw):,} chars, {len(results)} topics")
    return {**state, "raw_research": raw, "status": "research_complete",
            "errors": errors, "workflow_path": state.get("workflow_path", []) + ["research"]}


# ── Node 3: Embed & Store ─────────────────────────────────────────────────────
def embed_and_store_node(state: AgentState) -> AgentState:
    company   = state["company_name"]
    ticket    = state["ticket_id"]
    research  = state.get("raw_research", "")
    errors    = list(state.get("errors", []))
    namespace = company.lower().replace(" ", "-")
    chunks    = chunk_text(research)
    print(f"[EMBED] {len(chunks)} chunks → namespace '{namespace}'")

    vectors, ids, valid = [], [], []
    for i, chunk in enumerate(chunks):
        for attempt in range(3):
            try:
                emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[chunk])
                cid = f"{ticket}-chunk-{i:04d}"
                vectors.append({
                    "id": cid, "values": emb.data[0].embedding,
                    "metadata": {"company": company, "ticket_id": ticket,
                                 "text": chunk[:1000], "chunk_idx": i,
                                 "created_at": datetime.now(timezone.utc).isoformat()}
                })
                ids.append(cid); valid.append(chunk); break
            except Exception as e:
                if attempt == 2: errors.append(f"Embed failed chunk {i}: {e}")
                else: time.sleep(2 ** attempt)

    if not vectors:
        return {**state, "status": "error_embed_failed",
                "errors": errors + ["No vectors created"],
                "workflow_path": state.get("workflow_path", []) + ["embed_and_store"]}

    for i in range(0, len(vectors), 100):
        pinecone_index.upsert(vectors=vectors[i:i+100], namespace=namespace)

    print(f"[EMBED] ✅ {len(vectors)} vectors upserted")
    return {**state, "research_chunks": valid, "pinecone_ids": ids,
            "status": "stored", "errors": errors,
            "workflow_path": state.get("workflow_path", []) + ["embed_and_store"]}


# ── Node 4: Retrieve ──────────────────────────────────────────────────────────
def retrieve_node(state: AgentState) -> AgentState:
    company   = state["company_name"]
    namespace = company.lower().replace(" ", "-")
    errors    = list(state.get("errors", []))
    query     = f"{company} business model funding market competitors risks investment"
    print(f"[RETRIEVE] Querying Pinecone for '{company}'...")

    for attempt in range(3):
        try:
            emb    = openai_client.embeddings.create(model=EMBED_MODEL, input=[query])
            result = pinecone_index.query(
                vector=emb.data[0].embedding, top_k=10,
                namespace=namespace, include_metadata=True)
            chunks = [m["metadata"]["text"] for m in result["matches"]
                      if m.get("metadata", {}).get("text")]
            if not chunks:
                return {**state, "status": "error_no_data",
                        "errors": errors + ["No data in Pinecone"],
                        "workflow_path": state.get("workflow_path", []) + ["retrieve"]}
            print(f"[RETRIEVE] ✅ {len(chunks)} chunks")
            return {**state, "retrieved_chunks": chunks, "status": "retrieved",
                    "errors": errors, "workflow_path": state.get("workflow_path", []) + ["retrieve"]}
        except Exception as e:
            if attempt == 2:
                return {**state, "status": "error_retrieve_failed",
                        "errors": errors + [str(e)],
                        "workflow_path": state.get("workflow_path", []) + ["retrieve"]}
            time.sleep(2 ** attempt)


# ── Node 5: Rerank ────────────────────────────────────────────────────────────
def rerank_node(state: AgentState) -> AgentState:
    company = state["company_name"]
    chunks  = state.get("retrieved_chunks", [])
    errors  = list(state.get("errors", []))
    query   = f"Due diligence: {company} business model market risks investment recommendation"
    print(f"[RERANK] Reranking {len(chunks)} chunks...")

    for attempt in range(3):
        try:
            r = cohere_client.rerank(
                model="rerank-english-v3.0", query=query,
                documents=chunks, top_n=3, return_documents=True)
            reranked = [res.document.text for res in r.results if res.document]
            scores   = [round(res.relevance_score, 3) for res in r.results]
            print(f"[RERANK] ✅ Top-3 | Scores: {scores}")
            return {**state, "reranked_chunks": reranked, "status": "reranked",
                    "errors": errors, "workflow_path": state.get("workflow_path", []) + ["rerank"]}
        except Exception as e:
            if attempt == 2:
                print(f"[RERANK] ⚠️ Fallback to top-3 Pinecone results")
                return {**state, "reranked_chunks": chunks[:3], "status": "reranked_fallback",
                        "errors": errors + [f"Cohere failed: {e}"],
                        "workflow_path": state.get("workflow_path", []) + ["rerank"]}
            time.sleep(2 ** attempt)


# ── Node 6: Analyse ───────────────────────────────────────────────────────────
def analyse_node(state: AgentState) -> AgentState:
    company = state["company_name"]
    chunks  = state.get("reranked_chunks", [])
    errors  = list(state.get("errors", []))
    context = "\n\n---\n\n".join(chunks)
    print(f"[ANALYSE] Classifying '{company}'...")

    for attempt in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=RESEARCH_MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content":
                        f"Analyse {company}. Return JSON: risk_score (low/medium/high), "
                        f"opportunity_score (1-10 int), stage, sector, "
                        f"investment_priority (pass/watch/invest), key_strength, key_concern, confidence."},
                    {"role": "user", "content": f"Context:\n{context}"}
                ])
            parsed = json.loads(resp.choices[0].message.content)
            risk   = parsed.get("risk_score", "medium")
            opp    = int(parsed.get("opportunity_score", 5))
            print(f"[ANALYSE] ✅ Risk: {risk} | Opp: {opp}/10 | Priority: {parsed.get('investment_priority')}")
            return {**state, "risk_score": risk, "opportunity_score": opp,
                    "status": "analysed", "errors": errors,
                    "report_sections": {"_classification": json.dumps(parsed)},
                    "workflow_path": state.get("workflow_path", []) + ["analyse"]}
        except Exception as e:
            if attempt == 2:
                return {**state, "status": "error_analysis_failed",
                        "errors": errors + [str(e)],
                        "workflow_path": state.get("workflow_path", []) + ["analyse"]}
            time.sleep(2 ** attempt)


# ── Node 7: Generate Report ───────────────────────────────────────────────────
def generate_report_node(state: AgentState) -> AgentState:
    company  = state["company_name"]
    chunks   = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    errors   = list(state.get("errors", []))
    existing = state.get("report_sections") or {}
    context  = "\n\n---\n\n".join(chunks) if chunks else "No research context available."
    print(f"[REPORT] Generating 6-section report for '{company}'...")

    classification = {}
    if "_classification" in existing:
        try: classification = json.loads(existing["_classification"])
        except Exception: pass

    sections = dict(existing)
    for key, title in REPORT_SECTIONS:
        print(f"  Generating: {title}...")
        prompt = SECTION_PROMPTS[key].format(company=company)
        for attempt in range(3):
            try:
                resp = openai_client.chat.completions.create(
                    model=REPORT_MODEL, temperature=0.2,
                    messages=[
                        {"role": "system", "content":
                            f"You are a senior VC analyst writing a due diligence report on {company}. "
                            f"Use ONLY the provided research context. Be specific and factual. "
                            f"Risk: {state.get('risk_score','medium')} | "
                            f"Opportunity: {state.get('opportunity_score',5)}/10"},
                        {"role": "user", "content": f"Research:\n{context}\n\n{prompt}"}
                    ])
                sections[key] = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt == 2:
                    sections[key] = f"[Generation failed: {e}]"
                    errors.append(f"Section '{title}' failed: {e}")
                else:
                    time.sleep(2 ** attempt)

    complete = all(k in sections for k, _ in REPORT_SECTIONS)
    print(f"[REPORT] ✅ {sum(1 for k,_ in REPORT_SECTIONS if k in sections)}/6 sections")
    return {**state, "report_sections": sections, "report_ready": complete,
            "status": "report_generated", "errors": errors,
            "workflow_path": state.get("workflow_path", []) + ["generate_report"]}


# ── Node 8: Save Report ───────────────────────────────────────────────────────
def save_report_node(state: AgentState) -> AgentState:
    company  = state["company_name"]
    ticket   = state["ticket_id"]
    sections = state.get("report_sections", {})
    errors   = list(state.get("errors", []))

    # Save as markdown locally
    import pathlib
    md  = [f"# Due Diligence: {company}", f"**Ticket:** {ticket}",
           f"**Risk:** {state.get('risk_score')} | **Opportunity:** {state.get('opportunity_score')}/10", ""]
    for key, title in REPORT_SECTIONS:
        md.append(f"## {title}\n{sections.get(key,'_Not generated_')}\n")
    fname = f"report_{company.lower().replace(' ','_')}_{ticket}.md"
    pathlib.Path(fname).write_text("\n".join(md))
    print(f"[SAVE] ✅ Saved: {fname}")

    notion_url = None
    if NOTION_TOKEN and NOTION_DB_ID:
        try:
            import urllib.request
            payload = {
                "parent": {"database_id": NOTION_DB_ID},
                "properties": {
                    "Name":        {"title": [{"text": {"content": f"Due Diligence: {company}"}}]},
                    "Company":     {"rich_text": [{"text": {"content": company}}]},
                    "Ticket ID":   {"rich_text": [{"text": {"content": ticket}}]},
                    "Risk Score":  {"select": {"name": state.get("risk_score", "medium")}},
                    "Opportunity": {"number": state.get("opportunity_score", 5)},
                    "Status":      {"select": {"name": "Complete"}},
                },
                "children": [{"object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {
                        "content": sections.get("executive_summary", "")[:2000]}}]}}]
            }
            req = urllib.request.Request(
                "https://api.notion.com/v1/pages",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {NOTION_TOKEN}",
                         "Content-Type": "application/json",
                         "Notion-Version": "2022-06-28"},
                method="POST")
            with urllib.request.urlopen(req) as resp:
                data       = json.loads(resp.read())
                notion_url = data.get("url", "")
                print(f"[SAVE] ✅ Notion: {notion_url}")
        except Exception as e:
            errors.append(f"Notion save failed: {e}")
            print(f"[SAVE] ⚠️ Notion failed: {e}")

    return {**state,
            "notion_url": notion_url or f"file://{fname}",
            "status": "complete", "errors": errors,
            "workflow_path": state.get("workflow_path", []) + ["save_report"]}


# ── Error handler ─────────────────────────────────────────────────────────────
def error_handler_node(state: AgentState) -> AgentState:
    print(f"\n🚨 PIPELINE ERROR: {state.get('status')}")
    for e in state.get("errors", []): print(f"  ❌ {e}")
    return {**state, "status": "failed",
            "workflow_path": state.get("workflow_path", []) + ["error_handler"]}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD LANGGRAPH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def route(state: AgentState) -> str:
    return "error_handler" if "error" in state.get("status", "") else "continue"


def build_pipeline():
    g = StateGraph(AgentState)
    nodes = [
        ("validate_input",  validate_input_node),
        ("research",        research_node),
        ("embed_and_store", embed_and_store_node),
        ("retrieve",        retrieve_node),
        ("rerank",          rerank_node),
        ("analyse",         analyse_node),
        ("generate_report", generate_report_node),
        ("save_report",     save_report_node),
        ("error_handler",   error_handler_node),
    ]
    for name, fn in nodes:
        g.add_node(name, fn)

    g.set_entry_point("validate_input")

    steps = [n for n, _ in nodes if n != "error_handler"]
    for i, step in enumerate(steps[:-1]):
        g.add_conditional_edges(step, route,
            {"continue": steps[i+1], "error_handler": "error_handler"})

    g.add_conditional_edges("save_report", route,
        {"continue": END, "error_handler": "error_handler"})
    g.add_edge("error_handler", END)
    return g.compile()


print("Building LangGraph pipeline...")
pipeline = build_pipeline()
print("✅ LangGraph pipeline ready — 8 nodes")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Company Research Agent",
    description="Autonomous due diligence report generation powered by LangGraph",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    company_name: str


class ResearchResponse(BaseModel):
    ticket_id:         str
    company_name:      str
    status:            str
    risk_score:        Optional[str]
    opportunity_score: Optional[int]
    report_sections:   Optional[Dict[str, str]]
    notion_url:        Optional[str]
    errors:            List[str]
    workflow_path:     List[str]
    report_ready:      bool


@app.get("/health")
def health():
    """Health check — n8n can call this to verify server is running."""
    return {"status": "ok", "pipeline": "ready", "model": RESEARCH_MODEL}


@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    """
    Main endpoint — receives company name, runs full LangGraph pipeline,
    returns structured due diligence report.
    Called by n8n 'Call LangGraph Agent' HTTP Request node.
    """
    company = request.company_name.strip()
    if not company:
        raise HTTPException(status_code=400, detail="company_name is required")

    print(f"\n{'='*55}")
    print(f"  NEW REQUEST: {company}")
    print(f"{'='*55}")

    initial_state = AgentState(
        company_name=company, ticket_id="", initiated_at="",
        status="pending", errors=[],
        raw_research=None, research_chunks=None, pinecone_ids=None,
        retrieved_chunks=None, reranked_chunks=None,
        risk_score=None, opportunity_score=None,
        retry_count=0, report_sections=None,
        notion_url=None, report_ready=False, workflow_path=[]
    )

    result = pipeline.invoke(initial_state)

    print(f"\n✅ COMPLETE: {result.get('status')} | "
          f"Risk: {result.get('risk_score')} | "
          f"Opp: {result.get('opportunity_score')}/10")

    return ResearchResponse(
        ticket_id         = result.get("ticket_id", ""),
        company_name      = result.get("company_name", ""),
        status            = result.get("status", ""),
        risk_score        = result.get("risk_score"),
        opportunity_score = result.get("opportunity_score"),
        report_sections   = result.get("report_sections"),
        notion_url        = result.get("notion_url"),
        errors            = result.get("errors", []),
        workflow_path     = result.get("workflow_path", []),
        report_ready      = result.get("report_ready", False)
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUN SERVER
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀 Company Research Agent Server")
    print("  URL:    http://localhost:8000")
    print("  Docs:   http://localhost:8000/docs")
    print("  Health: http://localhost:8000/health")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

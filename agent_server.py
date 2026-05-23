

"""

agent_server.py

FastAPI server that wraps the LangGraph research pipeline.

n8n calls this server via HTTP POST /research.



Run with:

    python agent_server.py



Then test with:

    curl -X POST http://localhost:8000/research \

      -H "Content-Type: application/json" \

      -d '{"company_name": "Anthropic"}'

"""



import os, json, re, time, uuid

from pathlib import Path

from datetime import datetime, timezone

from typing import Any, Optional, List, Dict



from fastapi import FastAPI, HTTPException, Request

from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import uvicorn

from dotenv import load_dotenv



# ââ Load environment variables ââââââââââââââââââââââââââââââââââââââââââââââââ

load_dotenv()



OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY",   "")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

COHERE_API_KEY   = os.getenv("COHERE_API_KEY",   "")

NOTION_TOKEN     = os.getenv("NOTION_TOKEN",     "")

NOTION_DB_ID     = os.getenv("NOTION_DB_ID",     "")

PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "").strip()

REPORT_OUTPUT_DIR = Path(__file__).resolve().parent / ".generated_reports"

REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)





# ââ Import AI clients âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

from openai import OpenAI

from pinecone import Pinecone, ServerlessSpec

import cohere

from langgraph.graph import StateGraph, END

from typing import TypedDict



_openai_client = None

_pinecone_client = None

_cohere_client = None

_pinecone_index = None



PINECONE_INDEX_NAME = "company-research-agent"

EMBED_MODEL         = "text-embedding-3-small"

RESEARCH_MODEL      = "gpt-4o-mini"

REPORT_MODEL        = "gpt-4o"



def get_openai_client():

    global _openai_client

    if _openai_client is None:

        if not OPENAI_API_KEY:

            raise RuntimeError("OPENAI_API_KEY is not set")

        _openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return _openai_client





def get_pinecone_client():

    global _pinecone_client

    if _pinecone_client is None:

        if not PINECONE_API_KEY:

            raise RuntimeError("PINECONE_API_KEY is not set")

        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

    return _pinecone_client





def get_cohere_client():

    global _cohere_client

    if _cohere_client is None:

        if not COHERE_API_KEY:

            raise RuntimeError("COHERE_API_KEY is not set")

        _cohere_client = cohere.Client(COHERE_API_KEY)

    return _cohere_client



# ââ Pinecone index ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def get_or_create_index():

    pc = get_pinecone_client()

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

        print("â Pinecone index created")

    return pc.Index(PINECONE_INDEX_NAME)





def get_pinecone_index():

    global _pinecone_index

    if _pinecone_index is None:

        _pinecone_index = get_or_create_index()

        print("? Pinecone index ready")

    return _pinecone_index



# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

# AGENT STATE

# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class AgentState(TypedDict):

    company_name:      str

    ticket_id:         str

    initiated_at:      str

    public_base_url:   Optional[str]

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

    report_url:        Optional[str]

    report_ready:      bool

    token_usage:       Optional[Dict[str, Any]]

    estimated_cost_usd: Optional[float]

    cost_breakdown:     Optional[Dict[str, Any]]

    workflow_path:     List[str]





# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

# PIPELINE NODES

# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ



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



def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:

    words = text.split()

    step  = chunk_size - overlap

    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), step) if words[i:i+chunk_size]]





def normalize_company_slug(company_name: str) -> str:

    """

    Convert a company name into a safe file-name fragment.

    """

    slug = re.sub(r"[^a-z0-9]+", "_", company_name.lower()).strip("_")

    return slug or "company"





def append_workflow_step(state: AgentState, step: str) -> List[str]:

    return state.get("workflow_path", []) + [step]



def _normalize_token_usage(usage: Any) -> Dict[str, int]:

    if usage is None:

        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = getattr(usage, "input_tokens", None)

    output_tokens = getattr(usage, "output_tokens", None)

    if input_tokens is None:

        input_tokens = getattr(usage, "prompt_tokens", 0) or 0

    if output_tokens is None:

        output_tokens = getattr(usage, "completion_tokens", 0) or 0

    total_tokens = getattr(usage, "total_tokens", None)

    if total_tokens is None:

        total_tokens = int(input_tokens) + int(output_tokens)

    return {

        "input_tokens": int(input_tokens or 0),

        "output_tokens": int(output_tokens or 0),

        "total_tokens": int(total_tokens or 0),

    }



def record_token_usage(state: AgentState, step: str, usage: Any) -> Dict[str, Any]:

    current = dict(state.get("token_usage") or {})

    current.setdefault("input_tokens", 0)

    current.setdefault("output_tokens", 0)

    current.setdefault("total_tokens", 0)

    current.setdefault("steps", {})

    counts = _normalize_token_usage(usage)

    current["input_tokens"] += counts["input_tokens"]

    current["output_tokens"] += counts["output_tokens"]

    current["total_tokens"] += counts["total_tokens"]

    steps = dict(current.get("steps") or {})

    step_counts = dict(steps.get(step) or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    step_counts["input_tokens"] += counts["input_tokens"]

    step_counts["output_tokens"] += counts["output_tokens"]

    step_counts["total_tokens"] += counts["total_tokens"]

    steps[step] = step_counts

    current["steps"] = steps

    return current



def estimate_report_cost(state: AgentState) -> Dict[str, Any]:

    token_usage = state.get("token_usage") or {}

    steps = token_usage.get("steps") or {}

    breakdown: Dict[str, Dict[str, Any]] = {}

    total_cost = 0.0

    def add_step(
        key: str,
        input_rate: float,
        output_rate: float,
        *,
        tool_call_cost: float = 0.0,
        tool_calls: int = 0,
        extra_label: Optional[str] = None,
    ) -> None:

        nonlocal total_cost

        usage = steps.get(key) or {}

        input_tokens = int(usage.get("input_tokens", 0) or 0)

        output_tokens = int(usage.get("output_tokens", 0) or 0)

        step_cost = (
            (input_tokens / 1_000_000.0) * input_rate
            + (output_tokens / 1_000_000.0) * output_rate
            + (tool_call_cost * tool_calls)
        )

        breakdown[key] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens)),
            "tool_calls": tool_calls,
            "cost_usd": round(step_cost, 6),
        }

        if extra_label:
            breakdown[key]["label"] = extra_label

        total_cost += step_cost

    add_step("validate_input", 0.15, 0.60, extra_label="gpt-4o-mini")
    add_step(
        "research",
        0.15,
        0.60,
        tool_call_cost=0.025,
        tool_calls=len(RESEARCH_TOPICS),
        extra_label="gpt-4o-mini + web_search_preview",
    )
    add_step("embed_and_store", 0.02, 0.0, extra_label="text-embedding-3-small")
    add_step("analyse", 0.15, 0.60, extra_label="gpt-4o-mini")
    add_step("generate_report", 2.50, 10.00, extra_label="gpt-4o")

    return {
        "estimated_cost_usd": round(total_cost, 6),
        "cost_breakdown": breakdown,
    }





def is_openai_insufficient_quota_error(exc: Exception) -> bool:

    """

    Detect OpenAI billing/quota failures so we can stop retrying immediately.

    """

    message = str(exc).lower()

    code = getattr(exc, "code", None)

    status_code = getattr(exc, "status_code", None)

    return (

        code == "insufficient_quota"

        or status_code == 429 and "exceeded your current quota" in message

        or "insufficient_quota" in message

    )





def quota_error_state(state: AgentState, errors: List[str], step: str, exc: Exception) -> AgentState:

    return {

        **state,

        "status": f"error_openai_quota_exceeded_{step}",

        "errors": errors + [

            f"OpenAI quota exceeded during {step}: {exc}. Check billing/credits for the API key."

        ],

        "workflow_path": state.get("workflow_path", []) + [step],

    }





def resolve_public_base_url(request: Optional[Request] = None) -> str:

    """

    Resolve the externally reachable base URL for generated links.



    Priority:

    1. PUBLIC_BASE_URL from the environment

    2. Proxy headers / request host

    3. Localhost fallback

    """

    if PUBLIC_BASE_URL:

        return PUBLIC_BASE_URL.rstrip("/")



    if request is not None:

        forwarded_proto = request.headers.get("x-forwarded-proto")

        forwarded_host = request.headers.get("x-forwarded-host")

        host = forwarded_host or request.headers.get("host")

        if host:

            scheme = forwarded_proto or request.url.scheme or "http"

            return f"{scheme}://{host}".rstrip("/")



    return "http://localhost:8000"





# ââ Node 1: Validate ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def validate_input_node(state: AgentState) -> AgentState:

    raw    = state.get("company_name", "").strip()

    errors = list(state.get("errors", []))

    print(f"[VALIDATE] Raw input: '{raw}'")



    if not raw or len(raw) < 2:

        return {**state, "status": "error_invalid_input",

                "errors": errors + ["Invalid company name"],

                "workflow_path": state.get("workflow_path", []) + ["validate"]}



    # ââ Auto-correct typos using OpenAI ââââââââââââââââââââââââââââââââââââââ

    try:

        correction = get_openai_client().chat.completions.create(

            model="gpt-4o-mini",

            temperature=0,

            messages=[

                {

                    "role": "system",

                    "content": (

                        "You are a company name corrector. "

                        "The user may have made a typo in a company name. "

                        "Return ONLY the corrected company name â nothing else. "

                        "If the name looks correct already, return it unchanged. "

                        "Examples:\n"

                        "- 'Globacom Nieria' â 'Globacom Nigeria'\n"

                        "- 'Amazn' â 'Amazon'\n"

                        "- 'Anthropic' â 'Anthropic'\n"

                        "- 'Bloombeg' â 'Bloomberg'\n"

                        "Return ONLY the company name, no explanation."

                    )

                },

                {"role": "user", "content": raw}

            ]

        )

        corrected = correction.choices[0].message.content.strip()

        token_usage = record_token_usage(state, "validate_input", getattr(correction, "usage", None))



        if corrected.lower() != raw.lower():

            print(f"[VALIDATE] â Corrected: '{raw}' â '{corrected}'")

        else:

            print(f"[VALIDATE] â No correction needed: '{corrected}'")



    except Exception as e:

        if is_openai_insufficient_quota_error(e):

            return quota_error_state(state, errors, "validate", e)

        print(f"[VALIDATE] â ï¸ Correction failed, using original: {e}")

        corrected = " ".join(raw.split())



    ticket = f"RES-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"



    return {

        **state,

        "company_name":  corrected,

        "ticket_id":     ticket,

        "initiated_at":  datetime.now(timezone.utc).isoformat(),

        "status":        "validated",

        "errors":        errors,

        "token_usage":   token_usage if 'token_usage' in locals() else state.get("token_usage"),

        "workflow_path": state.get("workflow_path", []) + ["validate"]

    }



# ââ Node 2: Research ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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

                resp = get_openai_client().responses.create(

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

                    state = {**state, "token_usage": record_token_usage(state, "research", getattr(resp, "usage", None))}

                    break

            except Exception as e:

                if is_openai_insufficient_quota_error(e):

                    return quota_error_state(state, errors, "research", e)

                print(f"    â ï¸ Attempt {attempt+1}/3: {e}")

                if attempt == 2:

                    errors.append(f"Research topic failed: {e}")

                else:

                    time.sleep(2 ** attempt)



    if not results:

        return {**state, "status": "error_research_failed",

                "errors": errors + ["All research topics failed"],

                "workflow_path": state.get("workflow_path", []) + ["research"]}



    raw = f"# RESEARCH: {company}\n\n" + "\n\n".join(results)

    print(f"[RESEARCH] â {len(raw):,} chars, {len(results)} topics")

    return {**state, "raw_research": raw, "status": "research_complete",

            "errors": errors, "workflow_path": state.get("workflow_path", []) + ["research"]}





# ââ Node 3: Embed & Store âââââââââââââââââââââââââââââââââââââââââââââââââââââ

def embed_and_store_node(state: AgentState) -> AgentState:

    company   = state["company_name"]

    ticket    = state["ticket_id"]

    research  = state.get("raw_research", "")

    errors    = list(state.get("errors", []))

    namespace = company.lower().replace(" ", "-")

    chunks    = chunk_text(research)

    print(f"[EMBED] {len(chunks)} chunks â namespace '{namespace}'")



    vectors, ids, valid = [], [], []

    for i, chunk in enumerate(chunks):

        for attempt in range(3):

            try:

                emb = get_openai_client().embeddings.create(model=EMBED_MODEL, input=[chunk])

                state = {**state, "token_usage": record_token_usage(state, "embed_and_store", getattr(emb, "usage", None))}

                cid = f"{ticket}-chunk-{i:04d}"

                vectors.append({

                    "id": cid, "values": emb.data[0].embedding,

                    "metadata": {"company": company, "ticket_id": ticket,

                                 "text": chunk[:1000], "chunk_idx": i,

                                 "created_at": datetime.now(timezone.utc).isoformat()}

                })

                ids.append(cid); valid.append(chunk); break

            except Exception as e:

                if is_openai_insufficient_quota_error(e):

                    return quota_error_state(state, errors, "embed", e)

                if attempt == 2: errors.append(f"Embed failed chunk {i}: {e}")

                else: time.sleep(2 ** attempt)



    if not vectors:

        return {**state, "status": "error_embed_failed",

                "errors": errors + ["No vectors created"],

                "workflow_path": state.get("workflow_path", []) + ["embed_and_store"]}



    for i in range(0, len(vectors), 100):

        get_pinecone_index().upsert(vectors=vectors[i:i+100], namespace=namespace)



    print(f"[EMBED] â {len(vectors)} vectors upserted")

    return {**state, "research_chunks": valid, "pinecone_ids": ids,

            "status": "stored", "errors": errors,

            "workflow_path": state.get("workflow_path", []) + ["embed_and_store"]}





# ââ Node 4: Retrieve ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def retrieve_node(state: AgentState) -> AgentState:

    company   = state["company_name"]

    namespace = company.lower().replace(" ", "-")

    errors    = list(state.get("errors", []))

    query     = f"{company} business model funding market competitors risks investment"

    print(f"[RETRIEVE] Querying Pinecone for '{company}'...")



    for attempt in range(3):

        try:

            emb    = get_openai_client().embeddings.create(model=EMBED_MODEL, input=[query])

            result = get_pinecone_index().query(

                vector=emb.data[0].embedding, top_k=10,

                namespace=namespace, include_metadata=True)

            chunks = [m["metadata"]["text"] for m in result["matches"]

                      if m.get("metadata", {}).get("text")]

            if not chunks:

                return {**state, "status": "error_no_data",

                        "errors": errors + ["No data in Pinecone"],

                        "workflow_path": state.get("workflow_path", []) + ["retrieve"]}

            print(f"[RETRIEVE] â {len(chunks)} chunks")

            return {**state, "retrieved_chunks": chunks, "status": "retrieved",

                    "errors": errors, "workflow_path": state.get("workflow_path", []) + ["retrieve"]}

        except Exception as e:

            if is_openai_insufficient_quota_error(e):

                return quota_error_state(state, errors, "retrieve", e)

            if attempt == 2:

                return {**state, "status": "error_retrieve_failed",

                        "errors": errors + [str(e)],

                        "workflow_path": state.get("workflow_path", []) + ["retrieve"]}

            time.sleep(2 ** attempt)





# ââ Node 5: Rerank ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def rerank_node(state: AgentState) -> AgentState:

    company = state["company_name"]

    chunks  = state.get("retrieved_chunks", [])

    errors  = list(state.get("errors", []))

    query   = f"Due diligence: {company} business model market risks investment recommendation"

    print(f"[RERANK] Reranking {len(chunks)} chunks...")



    for attempt in range(3):

        try:

            r = get_cohere_client().rerank(

                model="rerank-english-v3.0", query=query,

                documents=chunks, top_n=3, return_documents=True)

            reranked = [res.document.text for res in r.results if res.document]

            scores   = [round(res.relevance_score, 3) for res in r.results]

            print(f"[RERANK] â Top-3 | Scores: {scores}")

            return {**state, "reranked_chunks": reranked, "status": "reranked",

                    "errors": errors, "workflow_path": state.get("workflow_path", []) + ["rerank"]}

        except Exception as e:

            if attempt == 2:

                print(f"[RERANK] â ï¸ Fallback to top-3 Pinecone results")

                return {**state, "reranked_chunks": chunks[:3], "status": "reranked_fallback",

                        "errors": errors + [f"Cohere failed: {e}"],

                        "workflow_path": state.get("workflow_path", []) + ["rerank"]}

            time.sleep(2 ** attempt)





# ââ Node 6: Analyse âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def analyse_node(state: AgentState) -> AgentState:

    company = state["company_name"]

    chunks  = state.get("reranked_chunks", [])

    errors  = list(state.get("errors", []))

    context = "\n\n---\n\n".join(chunks)

    print(f"[ANALYSE] Classifying '{company}'...")



    for attempt in range(3):

        try:

            resp = get_openai_client().chat.completions.create(

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

            state = {**state, "token_usage": record_token_usage(state, "analyse", getattr(resp, "usage", None))}

            risk   = parsed.get("risk_score", "medium")

            opp    = int(parsed.get("opportunity_score", 5))

            print(f"[ANALYSE] â Risk: {risk} | Opp: {opp}/10 | Priority: {parsed.get('investment_priority')}")

            return {**state, "risk_score": risk, "opportunity_score": opp,

                    "status": "analysed", "errors": errors,

                    "report_sections": {"_classification": json.dumps(parsed)},

                    "workflow_path": state.get("workflow_path", []) + ["analyse"]}

        except Exception as e:

            if is_openai_insufficient_quota_error(e):

                return quota_error_state(state, errors, "analyse", e)

            if attempt == 2:

                return {**state, "status": "error_analysis_failed",

                        "errors": errors + [str(e)],

                        "workflow_path": state.get("workflow_path", []) + ["analyse"]}

            time.sleep(2 ** attempt)





# ââ Node 7: Generate Report âââââââââââââââââââââââââââââââââââââââââââââââââââ

SECTION_PROMPTS = {

    "executive_summary":

        "Write a 150-word executive summary for {company}. "

        "Use this format:\n"

        "- What they do: [one sentence]\n"

        "- Founded: [year] | HQ: [location]\n"

        "- Stage: [stage] | Sector: [sector]\n"

        "- Key metric 1: [value]\n"

        "- Key metric 2: [value]\n"

        "- Key metric 3: [value]",



    "business_model":

        "Describe {company}'s business model. Use bullet points:\n"

        "Revenue Streams:\n"

        "- [stream 1]\n"

        "- [stream 2]\n"

        "Customer Segments:\n"

        "- [segment 1]\n"

        "Pricing Model:\n"

        "- [pricing detail]",



    "market_analysis":

        "Analyse the market for {company}. Use this format:\n"

        "Market Size:\n"

        "- TAM: [value]\n"

        "- SAM: [value]\n"

        "- SOM: [value]\n"

        "Growth:\n"

        "- [growth rate and trend]\n"

        "Key Trends:\n"

        "- [trend 1]\n"

        "- [trend 2]\n"

        "Regulatory Environment:\n"

        "- [key regulation]",



    "competitive_landscape":

        "Map the competitive landscape for {company}. Use this format:\n"

        "Top Competitors:\n"

        "- [Competitor 1]: [one line differentiation]\n"

        "- [Competitor 2]: [one line differentiation]\n"

        "- [Competitor 3]: [one line differentiation]\n"

        "Moat Strength: [Weak/Moderate/Strong]\n"

        "Key Differentiators:\n"

        "- [differentiator 1]\n"

        "- [differentiator 2]",



    "risk_factors":

        "List risk factors for {company}. Use this format:\n"

        "Technology Risk: [Low/Medium/High]\n"

        "- [explanation]\n"

        "Market Risk: [Low/Medium/High]\n"

        "- [explanation]\n"

        "Team Risk: [Low/Medium/High]\n"

        "- [explanation]\n"

        "Regulatory Risk: [Low/Medium/High]\n"

        "- [explanation]\n"

        "Financial Risk: [Low/Medium/High]\n"

        "- [explanation]",



    "investment_recommendation":

        "Write an investment recommendation for {company}. Use this format:\n"

        "Overall Score: [X]/10\n"

        "Rating: [Pass/Watch/Invest]\n\n"

        "Top Strengths:\n"

        "- [strength 1]\n"

        "- [strength 2]\n"

        "- [strength 3]\n\n"

        "Key Concerns:\n"

        "- [concern 1]\n"

        "- [concern 2]\n"

        "- [concern 3]\n\n"

        "Next Steps:\n"

        "- [step 1]\n"

        "- [step 2]",

}





def generate_report_node(state: AgentState) -> AgentState:

    company  = state["company_name"]

    chunks   = state.get("reranked_chunks") or state.get("retrieved_chunks") or []

    errors   = list(state.get("errors", []))

    existing = state.get("report_sections") or {}

    context  = "\n\n---\n\n".join(chunks) if chunks else "No research context available."

    print(f"[REPORT] Generating 6-section report for '{company}'...")



    sections = dict(existing)

    for key, title in REPORT_SECTIONS:

        print(f"  Generating: {title}...")

        prompt = SECTION_PROMPTS[key].format(company=company)

        for attempt in range(3):

            try:

                resp = get_openai_client().chat.completions.create(

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

                state = {**state, "token_usage": record_token_usage(state, "generate_report", getattr(resp, "usage", None))}

                break

            except Exception as e:

                if is_openai_insufficient_quota_error(e):

                    return quota_error_state(state, errors, "generate_report", e)

                if attempt == 2:

                    sections[key] = f"[Generation failed: {e}]"

                    errors.append(f"Section '{title}' failed: {e}")

                else:

                    time.sleep(2 ** attempt)



    complete = all(k in sections for k, _ in REPORT_SECTIONS)

    print(f"[REPORT] â {sum(1 for k, _ in REPORT_SECTIONS if k in sections)}/6 sections")

    return {**state, "report_sections": sections, "report_ready": complete,

            "status": "report_generated", "errors": errors,

            "workflow_path": state.get("workflow_path", []) + ["generate_report"]}





# ââ Node 8: Save Report âââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _save_report_node_legacy_stub(state: AgentState) -> AgentState:

    return state





def _send_telegram(token: str, chat_id: str, text: str):
    """
    Send a plain-text Telegram message.
    """
    import urllib.request

    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _send_telegram_document(token: str, chat_id: str, file_path: str, caption: str):
    """
    Upload a document to Telegram so it appears as a clickable file in chat.
    """
    import urllib.request



    boundary = "----CodexTelegramBoundary"

    with open(file_path, "rb") as f:

        file_bytes = f.read()



    # Rebuild the binary file part manually so we do not corrupt the PDF bytes.

    body = b"".join([

        f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}\r\n".encode(),

        f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode(),

        f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; filename=\"{os.path.basename(file_path)}\"\r\nContent-Type: application/pdf\r\n\r\n".encode(),

        file_bytes,

        f"\r\n--{boundary}--\r\n".encode(),

    ])



    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def extract_candidate_domains(text: str, company_name: Optional[str] = None) -> List[str]:
    """
    Best-effort extraction of likely official company domains from research text
    plus a few heuristic fallbacks based on the company name.
    """
    if not text and not company_name:
        return []

    blacklist = {
        "linkedin.com", "facebook.com", "x.com", "twitter.com", "instagram.com",
        "youtube.com", "wikipedia.org", "crunchbase.com", "bloomberg.com",
        "reuters.com", "techcrunch.com", "forbes.com", "pitchbook.com",
        "axios.com", "reddit.com", "github.com", "medium.com", "substack.com",
        "openai.com",  # avoid false positives when researching competitors
    }

    def score_domain(domain: str, company_tokens: List[str]) -> tuple:
        cleaned = domain.lower().strip(".,);:]}\"'")
        base = cleaned.split(":")[0]
        root = base[4:] if base.startswith("www.") else base
        score = 100
        if any(token and token in root for token in company_tokens):
            score -= 60
        if root.endswith(".com"):
            score -= 10
        if root.endswith(".ai"):
            score -= 6
        if root.count(".") <= 1:
            score -= 5
        if root in blacklist:
            score += 1000
        return (score, len(root), root)

    company_tokens = []
    if company_name:
        normalized = re.sub(r"[^a-z0-9]+", " ", company_name.lower()).strip()
        company_tokens = [token for token in normalized.split() if len(token) > 2]

    patterns = [
        r"https?://(?:www\.)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        r"\b(?:www\.)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text or "", flags=re.IGNORECASE))

    heuristic_roots = []
    if company_name:
        slug = re.sub(r"[^a-z0-9]+", "", company_name.lower())
        for suffix in (".com", ".ai", ".io", ".co"):
            heuristic_roots.append(f"{slug}{suffix}")

    ordered = []
    for domain in matches + heuristic_roots:
        cleaned = domain.lower().strip(".,);:]}\"'")
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)

    if not ordered and company_name:
        slug = re.sub(r"[^a-z0-9]+", "", company_name.lower())
        ordered = [f"{slug}.com", f"{slug}.ai", f"{slug}.io", f"{slug}.co"]

    ordered = sorted(
        [d for d in ordered if d not in blacklist],
        key=lambda d: score_domain(d, company_tokens),
    )
    return ordered


def fetch_logo_bytes(domain: str):
    """
    Try a couple of public logo sources. Returns (bytes, source_url) or (None, None).
    """
    if not domain:
        return None, None

    from urllib.request import Request, urlopen
    from urllib.parse import urljoin

    candidates = [
        f"https://logo.clearbit.com/{domain}?size=256",
        f"https://{domain}/favicon.ico",
        f"https://www.google.com/s2/favicons?domain={domain}&sz=256",
    ]

    for url in candidates:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=5) as resp:
                payload = resp.read()
            if payload:
                return payload, url
        except Exception:
            continue

    try:
        req = Request(f"https://{domain}", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8", "ignore")
        for pattern in (
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<link[^>]+rel=["\']icon["\'][^>]+href=["\']([^"\']+)["\']',
        ):
            match = re.search(pattern, html, flags=re.IGNORECASE)
            if match:
                image_url = match.group(1).strip()
                image_url = urljoin(f"https://{domain}", image_url)
                try:
                    req = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urlopen(req, timeout=5) as resp:
                        payload = resp.read()
                    if payload:
                        return payload, image_url
                except Exception:
                    pass
    except Exception:
        pass

    return None, None


def make_company_badge(company_name: str):
    """
    Create a simple vector monogram badge as a guaranteed fallback visual.
    """
    from reportlab.graphics.shapes import Drawing, Circle, String
    from reportlab.lib.colors import HexColor, white
    from reportlab.graphics import renderPM
    from reportlab.platypus import Image as RLImage
    from io import BytesIO

    letters = "".join(ch for ch in company_name.upper() if ch.isalnum())
    if not letters:
        letters = "CO"
    initials = letters[:2]

    size = 104
    drawing = Drawing(size, size)
    drawing.add(Circle(size / 2, size / 2, size / 2 - 2, fillColor=HexColor("#0B1F33"), strokeColor=HexColor("#0B1F33")))
    drawing.add(Circle(size / 2, size / 2, size / 2 - 11, fillColor=HexColor("#0F766E"), strokeColor=None))
    drawing.add(String(
        size / 2,
        size / 2 - 12,
        initials,
        fontName="Helvetica-Bold",
        fontSize=34 if len(initials) == 2 else 30,
        fillColor=white,
        textAnchor="middle",
    ))

    try:
        png_bytes = renderPM.drawToString(drawing, fmt="PNG")
        return RLImage(BytesIO(png_bytes), width=3.0 * cm, height=3.0 * cm)
    except Exception:
        return drawing


def save_report_node(state: AgentState) -> AgentState:
    company = state["company_name"]
    ticket = state["ticket_id"]
    sections = state.get("report_sections", {})
    errors = list(state.get("errors", []))
    raw_research = state.get("raw_research", "")

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from io import BytesIO
    from html import escape
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak

    section_titles = {
        "executive_summary": "Executive Summary",
        "business_model": "Business Model",
        "market_analysis": "Market Analysis",
        "competitive_landscape": "Competitive Landscape",
        "risk_factors": "Risk Factors",
        "investment_recommendation": "Investment Recommendation",
    }
    section_order = list(section_titles.items())

    report_path = REPORT_OUTPUT_DIR / f"report_{normalize_company_slug(company)}_{ticket}.pdf"
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        leftMargin=1.55 * cm,
        rightMargin=1.55 * cm,
        topMargin=2.0 * cm,
        bottomMargin=1.7 * cm,
    )

    page_width, page_height = A4
    styles = getSampleStyleSheet()
    brand_blue = HexColor("#0B1F33")
    brand_teal = HexColor("#0F766E")
    brand_soft = HexColor("#F4F7FB")
    line_color = HexColor("#DDE5EE")
    text_dark = HexColor("#182433")
    text_muted = HexColor("#607080")
    white_bg = HexColor("#FFFFFF")

    title_style = ParagraphStyle("CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=brand_blue, spaceAfter=4, tracking=0.2)
    company_style = ParagraphStyle("CompanyName", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=20.5, textColor=HexColor("#10263F"), spaceAfter=6)
    subtitle_style = ParagraphStyle("CoverSubtitle", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.9, leading=13.2, textColor=text_muted, spaceAfter=4)
    label_style = ParagraphStyle("Label", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.6, leading=9, textColor=HexColor("#7A8795"))
    value_style = ParagraphStyle("Value", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10.3, leading=11.6, textColor=text_dark)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13.8, textColor=text_dark, spaceAfter=4)
    bullet_style = ParagraphStyle("Bullet", parent=body_style, leftIndent=10, firstLineIndent=-5, bulletIndent=0, spaceAfter=2.2)
    section_header_style = ParagraphStyle("SectionHeader", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=14, textColor=white)
    footer_style = ParagraphStyle("Footer", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.2, leading=8.8, textColor=HexColor("#7A8795"), alignment=TA_CENTER)
    eyebrow_style = ParagraphStyle("Eyebrow", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=brand_teal)
    note_style = ParagraphStyle("Note", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10, textColor=HexColor("#7F8A98"), alignment=TA_CENTER)

    def make_metric_card(label, value, accent="#E8EEF5", value_color="#10263F"):
        card = Table([[Paragraph(escape(label), label_style), Paragraph(escape(value), ParagraphStyle("MetricValue", parent=value_style, textColor=HexColor(value_color), alignment=TA_RIGHT))]], colWidths=[4.1 * cm, 5.2 * cm])
        card.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), white_bg), ("BOX", (0, 0), (-1, -1), 0.8, line_color), ("LINEABOVE", (0, 0), (-1, 0), 2.0, HexColor(accent)), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return card

    def build_logo_candidate(company_name, raw_text):
        candidates = extract_candidate_domains(raw_text, company_name)
        for domain in candidates:
            logo_bytes, logo_source = fetch_logo_bytes(domain)
            if not logo_bytes:
                continue
            try:
                return RLImage(BytesIO(logo_bytes), width=2.35 * cm, height=2.35 * cm), logo_source
            except Exception:
                continue
        return make_company_badge(company_name), None

    logo_image, logo_source = build_logo_candidate(company, raw_research)
    risk = (state.get("risk_score") or "medium").lower()
    risk_palette = {"low": ("#16A34A", "#EAF7EF"), "medium": ("#D97706", "#FFF6E8"), "high": ("#DC2626", "#FDECEC")}
    risk_accent, risk_bg = risk_palette.get(risk, risk_palette["medium"])
    risk_label = risk.upper()
    opportunity = state.get("opportunity_score") or "N/A"
    token_usage = state.get("token_usage") or {}
    token_total = int(token_usage.get("total_tokens", 0) or 0)
    token_breakdown = token_usage.get("steps") or {}
    cost_info = estimate_report_cost(state)
    estimated_cost_usd = float(cost_info.get("estimated_cost_usd", 0.0) or 0.0)
    cost_breakdown = cost_info.get("cost_breakdown") or {}
    report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_time = datetime.now(timezone.utc).strftime("%H:%M UTC")

    def draw_page(canvas, doc_obj, first_page=False):
        canvas.saveState()
        canvas.setFillColor(brand_soft if first_page else white_bg)
        canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)
        canvas.setFillColor(brand_blue)
        canvas.rect(0, page_height - 0.88 * cm, page_width, 0.88 * cm, stroke=0, fill=1)
        canvas.setFillColor(HexColor("#D6E2F0"))
        canvas.rect(0, page_height - 0.93 * cm, page_width, 0.05 * cm, stroke=0, fill=1)
        canvas.setFont("Helvetica-Bold", 8.2)
        canvas.setFillColor(white)
        canvas.drawString(1.55 * cm, page_height - 0.56 * cm, f"{company}  |  {ticket}")
        canvas.setFont("Helvetica", 7.1)
        canvas.setFillColor(HexColor("#DCE7F5"))
        canvas.drawRightString(page_width - 1.55 * cm, page_height - 0.56 * cm, "Autonomous Company Research Agent")
        canvas.setStrokeColor(line_color)
        canvas.setLineWidth(0.5)
        canvas.line(1.55 * cm, 1.28 * cm, page_width - 1.55 * cm, 1.28 * cm)
        canvas.setFont("Helvetica", 7.0)
        canvas.setFillColor(HexColor("#7A8795"))
        canvas.drawString(1.55 * cm, 0.93 * cm, f"Generated {report_date} at {report_time}")
        canvas.drawRightString(page_width - 1.55 * cm, 0.93 * cm, f"Page {doc_obj.page}")
        canvas.restoreState()

    def cover_banner():
        banner = Table(
            [[
                Paragraph(
                    "AUTONOMOUS COMPANY RESEARCH BRIEF",
                    ParagraphStyle(
                        "BannerText",
                        parent=eyebrow_style,
                        fontName="Helvetica-Bold",
                        fontSize=8.2,
                        textColor=white,
                        alignment=TA_LEFT,
                        leading=10,
                    ),
                ),
                Paragraph(
                    "Live research | retrieval | reranking | report generation",
                    ParagraphStyle(
                        "BannerMeta",
                        parent=note_style,
                        fontName="Helvetica",
                        fontSize=7.3,
                        leading=9,
                        textColor=HexColor("#DCE7F5"),
                        alignment=TA_RIGHT,
                    ),
                ),
            ]],
            colWidths=[10.2 * cm, 6.4 * cm],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), brand_blue),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return banner

    def section_box(section_number, title, content, color_hex):
        out = []
        intro_map = {
            "Executive Summary": "High-level view of the company, momentum, and investing posture.",
            "Business Model": "How the company makes money, who pays, and what drives retention.",
            "Market Analysis": "Category context, growth signals, and market structure behind the story.",
            "Competitive Landscape": "Peers, substitutes, and the moat question in one place.",
            "Risk Factors": "The main downside risks that should shape diligence and sizing.",
            "Investment Recommendation": "The final call, confidence level, and next action.",
        }
        intro_text = intro_map.get(title, "Structured analysis and supporting context.")

        header = Table(
            [[
                Paragraph(
                    f"SECTION<br/><b>{section_number:02d}</b>",
                    ParagraphStyle(
                        "SectionNumber",
                        parent=section_header_style,
                        fontSize=8.3,
                        leading=9.5,
                        alignment=TA_CENTER,
                    ),
                ),
                Paragraph(
                    escape(title),
                    ParagraphStyle(
                        "SectionTitle",
                        parent=section_header_style,
                        fontSize=13,
                        leading=15,
                        alignment=TA_LEFT,
                    ),
                ),
            ]],
            colWidths=[2.15 * cm, 14.45 * cm],
        )
        header.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor(color_hex)),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        out.append(header)
        out.append(Spacer(1, 0.1 * cm))

        intro = Table(
            [[Paragraph(escape(intro_text), ParagraphStyle("SectionIntro", parent=note_style, fontSize=8.1, leading=10, textColor=HexColor("#5F6F81")))]],
            colWidths=[16.6 * cm],
        )
        intro.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F7FAFD")),
            ("BOX", (0, 0), (-1, -1), 0.7, line_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        out.append(intro)
        out.append(Spacer(1, 0.14 * cm))

        body_flow = []

        def add_paragraph(text, bullet=None):
            if bullet:
                body_flow.append(Paragraph(f"{bullet} {escape(text)}", bullet_style))
            else:
                body_flow.append(Paragraph(escape(text), body_style))

        for line in [line.strip() for line in content.splitlines() if line.strip()]:
            stripped = line.lstrip()
            if stripped.startswith(("- ", "* ", "• ")):
                add_paragraph(stripped[2:].strip(), bullet="•")
            elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ").":
                add_paragraph(stripped)
            elif stripped.endswith(":") and len(stripped) <= 70:
                body_flow.append(Paragraph(f"<b>{escape(stripped)}</b>", body_style))
            else:
                add_paragraph(stripped)

        out.append(Spacer(1, 0.02 * cm))
        if body_flow:
            out.extend(body_flow)
        else:
            out.append(Paragraph("No content generated for this section.", body_style))
        out.append(Spacer(1, 0.12 * cm))
        divider = Table([[""]], colWidths=[16.6 * cm])
        divider.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor(color_hex)),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ]))
        out.append(divider)
        return out

    def make_stat_chip(label, value, accent="#0B1F33"):
        chip = Table(
            [[
                Paragraph(escape(label), ParagraphStyle("ChipLabel", parent=label_style, fontSize=7.2, leading=8.5, textColor=HexColor("#708090"))),
                Paragraph(escape(value), ParagraphStyle("ChipValue", parent=value_style, fontSize=10.1, leading=11.4, textColor=HexColor(accent))),
            ]],
            colWidths=[2.25 * cm, 3.45 * cm],
        )
        chip.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FAFCFF")),
            ("BOX", (0, 0), (-1, -1), 0.7, HexColor("#DDE5EE")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return chip

    def make_logo_panel(image_obj, source_text=None):
        panel = [Spacer(1, 0.22 * cm), image_obj, Spacer(1, 0.22 * cm)]
        logo_table = Table([[panel]], colWidths=[5.65 * cm])
        logo_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), white_bg),
            ("BOX", (0, 0), (-1, -1), 1.0, line_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return logo_table

    elements = [cover_banner(), Spacer(1, 0.34 * cm)]
    left_panel = [
        Paragraph("Investment Due Diligence", title_style),
        Spacer(1, 0.1 * cm),
        Paragraph(escape(company), company_style),
        Paragraph(
            "A visually polished research pack generated from live web research, retrieval, reranking, and structured analysis.",
            subtitle_style,
        ),
        Spacer(1, 0.2 * cm),
        Table(
            [[
                make_stat_chip("Ticket", ticket, accent="#0B1F33"),
                make_stat_chip("Risk", risk_label, accent=risk_accent),
            ]],
            colWidths=[5.1 * cm, 5.1 * cm],
        ),
        Spacer(1, 0.1 * cm),
        Table(
            [[
                make_stat_chip("Opportunity", f"{opportunity}/10", accent="#0F766E"),
                make_stat_chip("Generated", report_date, accent="#4C6FFF"),
            ]],
            colWidths=[5.1 * cm, 5.1 * cm],
        ),
        Spacer(1, 0.1 * cm),
        Table(
            [[
                make_stat_chip("Tokens", f"{token_total:,}", accent="#111827"),
                make_stat_chip("Cost", f"${estimated_cost_usd:.4f}", accent="#059669"),
            ]],
            colWidths=[5.1 * cm, 5.1 * cm],
        ),
    ]
    if logo_image:
        right_panel = [make_logo_panel(logo_image, logo_source)]
    else:
        right_panel = [
            make_logo_panel(
                make_company_badge(company),
                "generated monogram fallback",
            )
        ]

    hero = Table([[left_panel, right_panel]], colWidths=[10.55 * cm, 6.05 * cm])
    hero.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), white_bg),
        ("BOX", (0, 0), (-1, -1), 1.0, line_color),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(hero)
    elements.append(Spacer(1, 0.28 * cm))
    summary_bar = Table(
        [[
            Paragraph(
                f"Prepared for <b>{escape(company)}</b>  |  {escape(ticket)}  |  Risk: <b>{escape(risk_label)}</b>  |  Opportunity: <b>{escape(str(opportunity))}/10</b>",
                ParagraphStyle(
                    "SummaryBar",
                    parent=body_style,
                    fontName="Helvetica-Bold",
                    fontSize=9.0,
                    leading=11,
                    textColor=white,
                    alignment=TA_CENTER,
                ),
            )
        ]],
        colWidths=[16.6 * cm],
    )
    summary_bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), brand_teal),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(summary_bar)
    elements.append(Spacer(1, 0.22 * cm))
    if token_breakdown:
        breakdown_text = ", ".join(
            f"{name}: {vals.get('total_tokens', 0):,}"
            for name, vals in token_breakdown.items()
        )
        elements.append(Paragraph(
            f"Token usage breakdown: {escape(breakdown_text)}",
            note_style,
        ))
        elements.append(Spacer(1, 0.08 * cm))
    if cost_breakdown:
        cost_text = ", ".join(
            f"{name}: ${vals.get('cost_usd', 0.0):.4f}"
            for name, vals in cost_breakdown.items()
        )
        elements.append(Paragraph(
            f"Cost breakdown: {escape(cost_text)}",
            note_style,
        ))
        elements.append(Spacer(1, 0.08 * cm))
    elements.append(Paragraph("This document is generated automatically. The first page is optimized for readability and the following pages are formatted for fast scanning.", note_style))
    elements.append(PageBreak())

    section_palette = {"executive_summary": "#2447D9", "business_model": "#0F766E", "market_analysis": "#7C3AED", "competitive_landscape": "#D97706", "risk_factors": "#DC2626", "investment_recommendation": "#059669"}
    for idx, (key, title) in enumerate(section_order, start=1):
        elements.extend(section_box(idx, title, sections.get(key, "Not generated."), section_palette.get(key, "#2447D9")))
        elements.append(Spacer(1, 0.22 * cm))

    elements.append(Spacer(1, 0.25 * cm))
    elements.append(Paragraph(f"Generated by Autonomous Research Agent for {escape(company)} | {escape(ticket)} | Risk: {escape(risk_label)} | Opportunity: {escape(str(opportunity))}/10", footer_style))
    doc.build(elements, onFirstPage=lambda c, d: draw_page(c, d, first_page=True), onLaterPages=draw_page)
    print(f"[SAVE] PDF report saved: {report_path}")

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    notion_url = None
    base_url = (state.get("public_base_url") or resolve_public_base_url()).rstrip("/")
    report_url = f"{base_url}/reports/{report_path.name}"
    pdf_url = f"file://{report_path.resolve()}"
    resolved_logo_source = logo_source or "generated monogram fallback"
    logo_found = bool(logo_source)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            summary = (
                f"Due Diligence Report Ready\n\n"
                f"Company: {company}\n"
                f"Ticket: {ticket}\n"
                f"Risk: {risk_label}\n"
                f"Opportunity: {opportunity}/10\n\n"
                f"The PDF is attached in this chat."
            )
            _send_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, summary)
            result = _send_telegram_document(
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
                str(report_path),
                caption=f"Due Diligence PDF: {company} | Ticket {ticket}\nSaved report: {report_path.name}",
            )
            if result.get("ok"):
                print("[SAVE] PDF sent to Telegram as a clickable document")
            else:
                print(f"[SAVE] Telegram PDF send failed: {result}")
        except Exception as e:
            print(f"[SAVE] Telegram send failed: {e}")

    return {
        **state,
        "notion_url": pdf_url,
        "report_url": report_url,
        "report_file_path": str(report_path.resolve()),
        "report_file_name": report_path.name,
        "logo_source": resolved_logo_source,
        "logo_found": logo_found,
        "status": "complete",
        "errors": errors,
        "token_usage": token_usage,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_breakdown": cost_breakdown,
        "workflow_path": append_workflow_step(state, "save_report"),
    }
# BUILD LANGGRAPH PIPELINE

# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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

    ]

    for name, fn in nodes:

        g.add_node(name, fn)



    # Error handler

    g.add_node("error_handler", lambda state: {

        **state, "status": "failed",

        "workflow_path": state.get("workflow_path", []) + ["error_handler"]

    })



    g.set_entry_point("validate_input")



    steps = [n for n, _ in nodes]

    for i, step in enumerate(steps[:-1]):

        g.add_conditional_edges(step, route,

            {"continue": steps[i+1], "error_handler": "error_handler"})



    g.add_conditional_edges("save_report", route,

        {"continue": END, "error_handler": "error_handler"})

    g.add_edge("error_handler", END)

    return g.compile()





print("Building LangGraph pipeline...")

pipeline = build_pipeline()

print("â LangGraph pipeline ready")



# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

# FASTAPI APP

# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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

    report_url:        Optional[str]

    report_file_path:  Optional[str]

    report_file_name:  Optional[str]

    logo_source:       Optional[str]

    logo_found:        Optional[bool]

    errors:            List[str]

    token_usage:       Optional[Dict[str, Any]]

    estimated_cost_usd: Optional[float]

    cost_breakdown:     Optional[Dict[str, Any]]

    workflow_path:     List[str]

    report_ready:      bool





@app.get("/health")

def health():

    """Health check â n8n can call this to verify server is running."""

    return {

        "status": "ok",

        "pipeline": "ready",

        "model": RESEARCH_MODEL,

        "public_base_url": resolve_public_base_url(),

    }





@app.get("/config")

def config():

    """Expose the resolved public URL so n8n or you can confirm it quickly."""

    return {

        "public_base_url": resolve_public_base_url(),

        "report_output_dir": str(REPORT_OUTPUT_DIR),

    }





@app.get("/reports/{filename}")

def download_report(filename: str):

    """

    Serve generated PDFs so n8n can download them as files.

    """

    safe_name = Path(filename).name

    report_path = REPORT_OUTPUT_DIR / safe_name

    if not report_path.exists() or report_path.suffix.lower() != ".pdf":

        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(

        path=str(report_path),

        media_type="application/pdf",

        filename=safe_name,

    )





@app.post("/research", response_model=ResearchResponse)

def run_research(request: ResearchRequest, http_request: Request):

    """

    Main endpoint â receives company name, runs full LangGraph pipeline,

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

        public_base_url=resolve_public_base_url(http_request),

        status="pending", errors=[],

        raw_research=None, research_chunks=None, pinecone_ids=None,

        retrieved_chunks=None, reranked_chunks=None,

        risk_score=None, opportunity_score=None,

        retry_count=0, report_sections=None,

        notion_url=None, report_url=None, report_file_path=None, report_file_name=None,
        logo_source=None, logo_found=None, report_ready=False,
        token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "steps": {}},
        estimated_cost_usd=0.0, cost_breakdown={},
        workflow_path=[]

    )



    result = pipeline.invoke(initial_state)



    print(f"\nâ COMPLETE: {result.get('status')} | "

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

        report_url        = result.get("report_url"),

        report_file_path  = result.get("report_file_path"),

        report_file_name  = result.get("report_file_name"),

        logo_source       = result.get("logo_source"),

        logo_found        = result.get("logo_found"),

        errors            = result.get("errors", []),

        token_usage       = result.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "steps": {}}),

        estimated_cost_usd = result.get("estimated_cost_usd", 0.0),

        cost_breakdown    = result.get("cost_breakdown", {}),

        workflow_path     = result.get("workflow_path", []),

        report_ready      = result.get("report_ready", False)

    )





# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

# RUN SERVER

# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

if __name__ == "__main__":

    port = int(os.getenv("PORT", "8000"))

    print("\n" + "="*55)

    print("  ð Company Research Agent Server")

    print(f"  URL:    http://0.0.0.0:{port}")

    print(f"  Docs:   http://0.0.0.0:{port}/docs")

    print(f"  Health: http://0.0.0.0:{port}/health")

    print("="*55 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)














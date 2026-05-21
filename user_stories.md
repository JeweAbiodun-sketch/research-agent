# User Stories

This document breaks the project into scrum-style user stories by day so the build progression is easy to follow.

## Sprint 1 - Day 1: Foundation and Input Validation

**Sprint Goal:** Make the input pipeline work end-to-end.

**User Story**
As an analyst, I can submit a company name and receive a structured ticket confirmation so I know the research has been initiated.

**Acceptance Criteria**
- The agent accepts a valid company name.
- The agent creates a ticket ID in the required format.
- The agent returns a structured acknowledgement quickly.
- Empty input is rejected with a clear error.

**Definition of Done**
- The validation and acknowledgement flow works end-to-end.
- The initial `AgentState` schema is ready for the later sprints.

## Sprint 2 - Day 2: Research Agent and Pinecone Storage

**Sprint Goal:** Research the company from the web and store the results in Pinecone.

**User Story**
As an analyst, the agent automatically researches the company from 3+ web sources and stores the results so I can query them later.

**Acceptance Criteria**
- The agent runs OpenAI web search across the required research topics.
- The research output is chunked into manageable pieces.
- The chunks are embedded and upserted to Pinecone.
- The stored vectors include the correct company metadata.

**Definition of Done**
- Pinecone contains at least 10 vectors for the researched company.
- The research output is persisted for downstream retrieval.

## Sprint 3 - Day 3: Retrieval, Reranking, and Analysis

**Sprint Goal:** Retrieve relevant research, rerank it, and classify the investment signals.

**User Story**
As an analyst, I want the agent to retrieve the most relevant research, rerank it for quality, and classify the company's risk and opportunity so I can make an informed decision faster.

**Acceptance Criteria**
- Pinecone returns the top matching chunks for the company query.
- Cohere reranks the retrieved chunks by relevance.
- The analysis step produces a risk score and an opportunity score.
- Errors are handled and reported clearly.

**Definition of Done**
- Cohere returns the top 3 reranked chunks.
- The LangGraph analysis node produces `risk_score`, `opportunity_score`, and investment priority.

## Sprint 4 - Day 4: Report Generation

**Sprint Goal:** Turn the reranked context into a structured investment report.

**User Story**
As an analyst, I want the agent to generate a 6-section due diligence report from the reranked research so I can review the company in a standard format.

**Acceptance Criteria**
- The report includes all 6 required sections.
- The content is generated from the research context.
- The report is written in a clear, investment-focused style.

**Definition of Done**
- All 6 report sections are generated successfully.
- The report is ready for saving.

## Sprint 5 - Day 5: Save and Deliver

**Sprint Goal:** Save the report and complete the full pipeline with a usable output.

**User Story**
As an analyst, I want the finished report saved to Notion or Markdown so I can access and share the result immediately.

**Acceptance Criteria**
- The report is saved to Notion when credentials are available.
- If Notion fails, the report is saved locally as Markdown.
- The full pipeline can run end-to-end without manual intervention.

**Definition of Done**
- The report is saved successfully in at least one supported format.
- The agent completes the workflow and returns a final status.

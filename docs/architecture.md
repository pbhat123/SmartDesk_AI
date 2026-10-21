# SmartDesk AI architecture

One request goes through the chat service (guards, limits), the LangGraph agent, the tools, the output guard and the ops layer.
State lives in a SQLite checkpointer keyed by `thread_id` only. The same diagram is exported as [`architecture.svg`](architecture.svg) and [`architecture.png`](architecture.png); this file is the single source (FR-I1 to I9, PLAN section 4).

```mermaid
flowchart TD
  UI["UI: Streamlit chat + Monitoring tab, CLI"] --> RL

  subgraph CS["ChatService"]
    direction TB
    RL["Rate limits, then budget cutoff"] --> IG["Input guard: PII masking, name reduction,<br/>self-harm and threat rules, Moderation<br/>(plain harassment answered calmly,<br/>illicit below 0.50 allowed), injection check"]
  end

  IG --> PR

  subgraph AG["LangGraph agent"]
    direction TB
    PR["prepare: email, compaction,<br/>long-term recall"] --> OR["Orchestrator: intent +<br/>standalone sub-questions"]
    OR -->|smalltalk, out of scope| CC["chitchat"]
    OR -->|kb_query| CL["cache_lookup"]
    OR -->|create_ticket, ticket_status| TA

    subgraph KBP["Knowledge path"]
      direction TB
      ITW["IT worker"]
      HRW["HR worker"]
      MW["no-domain search (MISC)"]
      RG["Retry gate: whole-KB retry once"]
      CO["Compose: pass-through for one part,<br/>Synthesizer for two or more,<br/>ticket offer for unanswered parts"]
      ITW --> RG
      HRW --> RG
      MW --> RG
      RG --> CO
    end

    subgraph TA["Ticket agent (sequential)"]
      direction TB
      TC["collect email, duplicate check, draft"] --> HI{"HITL approval gate<br/>interrupt: yes / no / edit"}
      HI -->|yes| TCR["create ticket"]
      SL["status flow: 0, 1 or 2+ tickets"]
    end

    CL -->|miss: one Send per part| ITW
    CL --> HRW
    CL --> MW
  end

  CC --> OG
  CL -->|hit| OG
  CO --> OG
  TCR --> OG
  SL --> OG

  ITW -.search_kb.-> T1
  HRW -.-> T1
  MW -.-> T1
  TCR -.create_ticket.-> T2
  SL -.get_tickets.-> T2

  subgraph TOOLS["Tool registry and executor"]
    direction LR
    T1["search_kb: BM25 + Chroma dense,<br/>RRF fusion, bge-reranker-base"]
    T2["GitHub REST tickets client"]
  end
  T1 --> KB[("KB: Markdown policies,<br/>Chroma + BM25 index")]
  T2 --> GH[("GitHub Issues (public repo)<br/>or in-memory mock")]
  LT[("Long-term memory: SQLite,<br/>per hashed email")] -.recall.-> PR
  TCR -.write after confirmed create.-> LT

  OG["Output guard: grounding check, Moderation,<br/>PII scan, schema validation"] --> REPLY["Reply to the user"]

  subgraph OPS["Ops"]
    direction LR
    CA["Answer cache: exact + semantic,<br/>Redis or in-memory"]
    SP["Spend ledger, budget"]
    TR["Traces: local log, Langfuse,<br/>LangSmith (masked)"]
    MO["Monitoring tab"]
    SP --> MO
    TR --> MO
  end
  REPLY -.-> OPS
```

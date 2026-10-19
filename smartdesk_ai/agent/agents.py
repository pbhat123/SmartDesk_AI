"""The agents of the multi-agent graph (FR-I1) and the versioned prompt each one loads (FR-H14).

The name is what a call is labelled with in the cost tracker, the LLM records, the tool metrics and the trace
(FR-H7, FR-H9, FR-G9, FR-I9). Plain constants only, so any module can import it.
"""
ORCHESTRATOR = "orchestrator"
IT_WORKER = "it_worker"
HR_WORKER = "hr_worker"
SYNTHESIZER = "synthesizer"
TICKET_AGENT = "ticket_agent"

AGENTS = (ORCHESTRATOR, IT_WORKER, HR_WORKER, SYNTHESIZER, TICKET_AGENT)

# Prompt name (the `prompts:` key in config.yaml and the file prefix in agent/prompts/) per agent.
AGENT_PROMPTS = {
    ORCHESTRATOR: "orchestrator",
    IT_WORKER: "worker_it",
    HR_WORKER: "worker_hr",
    SYNTHESIZER: "synthesizer",
    TICKET_AGENT: "ticket_summary",
}

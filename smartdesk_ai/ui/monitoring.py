"""The Monitoring tab: what the app has been doing, read from local files so it works with no keys (FR-G10).

Read-only. It never creates or changes a file: a missing log is an empty section. Sources:
- tool stats: `tool_metrics.jsonl` (FR-G9, G10; records hold no arguments, so no email)
- degraded modes: the health registry (FR-F7)
- cost by tier: the spend ledger (`data/spend_ledger.jsonl`, the audit trail to compare with the provider's bill)
  next to the request cost tracker's `cost.jsonl`; the daily budget from `ops.db` (FR-H7, H8)
- stale chunks: the KB hashes in the index state (FR-H6)
- agent metrics: the last `agent_metrics.json` an eval run saved (FR-H11)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.guardrails.pii import mask_emails_in
from smartdesk_ai.ops.budget import BudgetController, SpendStore
from smartdesk_ai.ops.cost import CostEntry
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ops.metrics import ToolCallRecord, load_records, summarize_tools
from smartdesk_ai.ops.spend_ledger import ledger_path, read_events
from smartdesk_ai.rag.embeddings import HashingEmbeddings, embedder_id
from smartdesk_ai.rag.staleness import StaleReport, check_staleness
from smartdesk_ai.ui.context import UiContext

TOOL_METRICS_FILE = "tool_metrics.jsonl"
COST_FILE = "cost.jsonl"
OPS_DB_FILE = "ops.db"
AGENT_METRICS_FILE = "agent_metrics.json"
TIER_ORDER = ("main", "light", "embedding", "other")


class ToolRow(BaseModel):
    tool: str
    calls: int
    success_rate: float
    failures: int
    failure_types: dict[str, int]
    retries: int  # attempts beyond the first, summed over calls
    retry_rate: float  # share of calls that needed more than one attempt
    p50_ms: float
    p95_ms: float


class DegradedRow(BaseModel):
    component: str
    mode: str
    reason: str
    since: str | None


class TierRow(BaseModel):
    tier: str
    models: list[str] = Field(default_factory=list)
    ledger_calls: int = 0
    ledger_usd: float = 0.0
    tracker_steps: int = 0
    tracker_usd: float = 0.0


class BudgetRow(BaseModel):
    spent_today_usd: float | None  # None when ops.db does not exist yet
    level: int
    cutoff_1_usd: float
    cutoff_2_usd: float


class AgentMetrics(BaseModel):
    """The numbers `evals/agent_metrics.py` saves (FR-H11)."""

    model_config = ConfigDict(extra="ignore")

    mock: bool = False
    n_tasks: int
    n_turns: int
    completion_rate: float | None = None
    task_accuracy: float | None = None
    steps_to_completion: float | None = None
    task_latency_ms: float | None = None
    turn_p50_ms: float | None = None
    turn_p95_ms: float | None = None
    usd_per_task: float | None = None
    error_rate: float | None = None


class MonitoringData(BaseModel):
    tools: list[ToolRow]
    degraded: list[DegradedRow]
    tiers: list[TierRow]
    ledger_total_usd: float
    ledger_today_usd: float
    budget: BudgetRow
    stale: StaleReport
    agent_metrics: AgentMetrics | None


def _tool_rows(path: Path) -> list[ToolRow]:
    records = load_records(path)
    retries: dict[str, int] = {}
    for rec in records:
        retries[rec.tool] = retries.get(rec.tool, 0) + max(rec.attempts - 1, 0)
    return [ToolRow(tool=s.tool, calls=s.calls, success_rate=s.success_rate, failures=sum(s.failures.values()),
                    failure_types=s.failures, retries=retries.get(s.tool, 0), retry_rate=s.retry_rate,
                    p50_ms=s.p50_ms, p95_ms=s.p95_ms) for s in summarize_tools(records)]


def _tier_of(model: str | None, settings: Settings) -> str:
    """The config tier a model id belongs to; a dated model name matches its configured id as a prefix."""
    known = {"main": settings.models.main.id, "light": settings.models.light.id,
             "embedding": settings.models.embedding.id}
    hits = [(len(mid), tier) for tier, mid in known.items() if model and model.startswith(mid)]
    return max(hits)[1] if hits else "other"


def _tracker_entries(path: Path) -> list[CostEntry]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        try:
            entries.append(CostEntry.model_validate_json(line))
        except ValidationError:
            continue  # a half-written last line
    return entries


def _tier_rows(settings: Settings, ledger: Path, tracker: Path) -> tuple[list[TierRow], float, float]:
    rows = {tier: TierRow(tier=tier) for tier in TIER_ORDER}
    today = datetime.now(timezone.utc).date().isoformat()  # the ledger counts days in UTC, as the billing page does
    total = today_usd = 0.0
    for event in read_events(ledger):
        if event.get("event") != "spend":
            continue
        row = rows[_tier_of(event.get("model"), settings)]
        row.ledger_calls += event.get("calls", 0)
        row.ledger_usd += event.get("usd", 0.0)
        if event.get("model") and event["model"] not in row.models:
            row.models.append(event["model"])
        total += event.get("usd", 0.0)
        if str(event.get("ts", ""))[:10] == today:
            today_usd += event.get("usd", 0.0)
    for entry in _tracker_entries(tracker):
        if entry.step == "tool":
            continue  # tool steps have no tier and no price
        row = rows[entry.tier if entry.step == "llm" and entry.tier in rows else
                   "embedding" if entry.step == "embedding" else "other"]
        row.tracker_steps += 1
        row.tracker_usd += entry.usd
    for row in rows.values():
        row.ledger_usd, row.tracker_usd = round(row.ledger_usd, 6), round(row.tracker_usd, 6)
    return list(rows.values()), round(total, 6), round(today_usd, 6)


def _budget_row(settings: Settings, ops_db: Path) -> BudgetRow:
    cfg = settings.budget
    if not ops_db.is_file():  # opening a SpendStore would create the file
        return BudgetRow(spent_today_usd=None, level=0, cutoff_1_usd=cfg.cutoff_1_usd, cutoff_2_usd=cfg.cutoff_2_usd)
    controller = BudgetController(settings, SpendStore(ops_db))
    return BudgetRow(spent_today_usd=round(controller.spend_today(), 6), level=controller.level(),
                     cutoff_1_usd=cfg.cutoff_1_usd, cutoff_2_usd=cfg.cutoff_2_usd)


def _agent_metrics(results_dir: Path) -> AgentMetrics | None:
    path = results_dir / AGENT_METRICS_FILE
    try:
        return AgentMetrics.model_validate(json.loads(path.read_text()))
    except (OSError, ValueError):  # missing, unreadable, malformed or the wrong shape: the empty state
        return None


def load_monitoring(
    settings: Settings, data_dir: Path, kb_dir: Path, registry: HealthRegistry, *,
    results_dir: Path, ledger: Path | None = None,
) -> MonitoringData:
    data_dir = Path(data_dir)
    live = settings.modes.llm == "live"
    index_dir = data_dir / ("index" if live else "index_mock")  # the folders `build_stack` uses
    embedder = settings.models.embedding.id if live else embedder_id(HashingEmbeddings())
    tiers, ledger_total, ledger_today = _tier_rows(settings, ledger or ledger_path(), data_dir / COST_FILE)
    return MonitoringData(
        tools=_tool_rows(data_dir / TOOL_METRICS_FILE),
        degraded=[DegradedRow(component=h.component.value, mode=h.mode or "", reason=mask_emails_in(h.reason or ""),
                              since=h.since.isoformat(timespec="seconds") if h.since else None)
                  for h in registry.degraded_modes()],
        tiers=tiers, ledger_total_usd=ledger_total, ledger_today_usd=ledger_today,
        budget=_budget_row(settings, data_dir / OPS_DB_FILE),
        stale=check_staleness(kb_dir, index_dir, settings.chunking, embedder),
        agent_metrics=_agent_metrics(Path(results_dir)),
    )


# --- rendering ------------------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _render_tools(data: MonitoringData) -> None:
    st.subheader("Tool stats")
    if not data.tools:
        st.caption("No tool calls yet.")
        return
    st.dataframe(pd.DataFrame([{
        "tool": r.tool, "calls": r.calls, "success rate": round(r.success_rate, 3), "failures": r.failures,
        "failed by": ", ".join(f"{k}: {v}" for k, v in sorted(r.failure_types.items())) or "-",
        "retries": r.retries, "retry rate": round(r.retry_rate, 3), "p50 ms": round(r.p50_ms, 1),
        "p95 ms": round(r.p95_ms, 1)} for r in data.tools]), hide_index=True, use_container_width=True)


def _render_degraded(data: MonitoringData) -> None:
    st.subheader("Degraded modes")
    if not data.degraded:
        st.caption("Every optional component is running normally.")
        return
    st.dataframe(pd.DataFrame([{"component": d.component, "running instead": d.mode, "reason": d.reason,
                                "since": d.since or ""} for d in data.degraded]),
                 hide_index=True, use_container_width=True)


def _render_cost(data: MonitoringData) -> None:
    st.subheader("Cost by tier")
    left, middle, right = st.columns(3)
    left.metric("Ledger, all time", f"${data.ledger_total_usd:.4f}")
    middle.metric("Ledger, today (UTC)", f"${data.ledger_today_usd:.4f}")
    spent = data.budget.spent_today_usd
    right.metric("Daily budget used", "no spend recorded" if spent is None else f"${spent:.4f}",
                 help=f"Cutoff 1 at ${data.budget.cutoff_1_usd:.2f} (Light-Model only), "
                      f"cutoff 2 at ${data.budget.cutoff_2_usd:.2f} (service stops). Level now: {data.budget.level}.")
    st.dataframe(pd.DataFrame([{
        "tier": t.tier, "models": ", ".join(t.models) or "-", "ledger calls": t.ledger_calls,
        "ledger USD": t.ledger_usd, "tracker steps": t.tracker_steps, "tracker USD": t.tracker_usd}
        for t in data.tiers]), hide_index=True, use_container_width=True)
    st.caption("The ledger is the audit trail to compare with the provider's billing page; the tracker is the "
               "per-request analytics. Mock runs price nothing, so both stay at 0 there. 'other' is a model "
               "missing from config.yaml (priced as the Main model).")


def _render_stale(data: MonitoringData) -> None:
    st.subheader("Stale KB chunks")
    stale = data.stale
    if stale.reason:
        st.warning(f"All chunks are stale: {stale.reason}. Re-run `python -m smartdesk_ai.rag.ingest`.")
    elif stale.stale_chunk_ids or stale.new_files:
        if stale.stale_chunk_ids:
            st.warning(f"{len(stale.stale_chunk_ids)} chunk(s) from changed or deleted files: "
                       f"{', '.join(stale.stale_files + stale.deleted_files)}. "
                       "Re-run `python -m smartdesk_ai.rag.ingest` to refresh them.")
            st.dataframe(pd.DataFrame({"stale chunk": stale.stale_chunk_ids}), hide_index=True,
                         use_container_width=True)
        if stale.new_files:
            st.info(f"Not yet ingested: {', '.join(stale.new_files)}.")
    else:
        st.caption("No stale chunks: every indexed file matches the hash stored at ingest.")


def _render_agent_metrics(data: MonitoringData) -> None:
    st.subheader("Agent metrics")
    m = data.agent_metrics
    if m is None:
        st.caption("No agent metrics yet. Create them with "
                   "`python -m evals.run --only agent --mock --out-dir evals/results`.")
        return
    st.caption(f"From the last saved eval run: {m.n_tasks} scripted tasks, {m.n_turns} turns"
               + (" (mock models, so cost is 0)." if m.mock else "."))
    cells = [("Task completion", _pct(m.completion_rate)), ("Task accuracy", _pct(m.task_accuracy)),
             ("Error rate", _pct(m.error_rate)),
             ("Steps to completion", "n/a" if m.steps_to_completion is None else f"{m.steps_to_completion:.1f}"),
             ("Task latency", "n/a" if m.task_latency_ms is None else f"{m.task_latency_ms:.0f} ms"),
             ("Turn latency p50 / p95", "n/a" if m.turn_p50_ms is None or m.turn_p95_ms is None
              else f"{m.turn_p50_ms:.0f} / {m.turn_p95_ms:.0f} ms"),
             ("Cost per task", "n/a" if m.usd_per_task is None else f"${m.usd_per_task:.4f}")]
    for start in range(0, len(cells), 4):
        for column, (label, value) in zip(st.columns(4), cells[start:start + 4]):
            column.metric(label, value)
    st.caption("Throughput (requests per second) is measured by `scripts/load_test.py`, not by this tab.")


def render_monitoring(ctx: UiContext) -> None:
    if st.button("Refresh", key="monitoring_refresh"):
        st.rerun()
    data = load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir)
    _render_tools(data)
    _render_degraded(data)
    _render_cost(data)
    _render_stale(data)
    _render_agent_metrics(data)

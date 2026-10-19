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
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.guardrails.pii import mask_emails_in
from smartdesk_ai.ops.budget import BudgetController, SpendStore
from smartdesk_ai.ops.cost import CostEntry
from smartdesk_ai.ops.drift import (
    DRIFT_LOG,
    DriftCeilingExceeded,
    DriftHistory,
    DriftPoint,
    DriftRunner,
    Golden,
    describe_estimate,
    estimate_drift,
    make_runner,
)
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ops.hitl_counters import HitlCounts, load_counts
from smartdesk_ai.ops.metrics import ToolCallRecord, date_from_trace, load_records, summarize_tools
from smartdesk_ai.ops.spend_ledger import ledger_path, read_events
from smartdesk_ai.rag.embeddings import HashingEmbeddings, embedder_id
from smartdesk_ai.rag.staleness import StaleReport, check_staleness
from smartdesk_ai.ui.context import UiContext

TOOL_METRICS_FILE = "tool_metrics.jsonl"
TRACE_FILE = "trace.jsonl"
COST_FILE = "cost.jsonl"
HITL_FILE = "hitl.jsonl"
OPS_DB_FILE = "ops.db"
AGENT_METRICS_FILE = "agent_metrics.json"
TIER_ORDER = ("main", "light", "embedding", "other")

ToolWindowName = Literal["app", "today", "all"]
HitlWindowName = Literal["session", "today", "all"]
TOOL_WINDOWS: dict[str, ToolWindowName] = {"Since app start": "app", "Today (UTC)": "today", "All time": "all"}
HITL_WINDOWS: dict[str, HitlWindowName] = {"This session": "session", "Today (UTC)": "today", "All time": "all"}
APP_STARTED = datetime.now(timezone.utc)  # set when the server process first imports this module


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
    last_error_ts: str | None = None  # UTC time of the tool's latest failed call in the window
    last_error_type: str | None = None  # its outcome: validation_error, unavailable or error


class ToolWindow(BaseModel):
    """The time span the tool stats cover."""

    label: str
    since: str | None  # the start of the window; for "All time" the first dated call
    first_ts: str | None
    last_ts: str | None
    calls: int
    undated: int = 0  # calls logged before times were recorded that the trace could not date (only in "All time")


class DegradedRow(BaseModel):
    component: str
    mode: str
    reason: str
    since: str | None


class ComponentRow(BaseModel):
    component: str
    status: str  # ok, degraded, skipped or unknown (not checked yet)
    running_instead: str  # the substitute in use while degraded
    detail: str  # why it is degraded or skipped
    load_seconds: float | None


class SourceRow(BaseModel):
    """Today's ledger spend of one kind of run: the app, an eval run, a script, a drift check."""

    source: str
    usd: float = 0.0
    calls: int = 0


class SessionCost(BaseModel):
    usd: float = 0.0
    llm_calls: int = 0
    tokens: int = 0


class TierRow(BaseModel):
    tier: str
    model: str = "-"  # the configured model id of the tier
    seen_as: list[str] = Field(default_factory=list)  # dated names the provider billed under it
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
    tool_window: ToolWindow
    components: list[ComponentRow]
    degraded: list[DegradedRow]
    tiers: list[TierRow]
    ledger_total_usd: float
    ledger_today_usd: float
    sources_today: list[SourceRow]
    budget: BudgetRow
    stale: StaleReport
    hitl: HitlCounts
    hitl_label: str = "All time"
    drift: list[DriftPoint]
    agent_metrics: AgentMetrics | None


def _parse(ts: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None
    return parsed if parsed is None or parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _day_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _tool_view(data_dir: Path, window: ToolWindowName, now: datetime, app_started: datetime) -> tuple[list[ToolRow], ToolWindow]:
    records = date_from_trace(load_records(data_dir / TOOL_METRICS_FILE), data_dir / TRACE_FILE)
    start = {"app": app_started, "today": _day_start(now), "all": None}[window]
    label = {"app": "Since app start", "today": "Today (UTC)", "all": "All time"}[window]
    undated = sum(r.ts is None for r in records)
    kept = [r for r in records if start is None or ((when := _parse(r.ts)) is not None and when >= start)]
    times = sorted((r.ts for r in kept if r.ts), key=lambda t: _parse(t))
    since = start.isoformat(timespec="seconds") if start is not None else (times[0] if times else None)
    info = ToolWindow(label=label, since=since, first_ts=times[0] if times else None,
                      last_ts=times[-1] if times else None, calls=len(kept), undated=undated if window == "all" else 0)
    retries: dict[str, int] = {}
    last_error: dict[str, ToolCallRecord] = {}
    for rec in kept:
        retries[rec.tool] = retries.get(rec.tool, 0) + max(rec.attempts - 1, 0)
        if rec.outcome != "success" and rec.ts and (rec.tool not in last_error
                                                    or _parse(rec.ts) >= _parse(last_error[rec.tool].ts)):
            last_error[rec.tool] = rec
    rows = [ToolRow(tool=s.tool, calls=s.calls, success_rate=s.success_rate, failures=sum(s.failures.values()),
                    failure_types=s.failures, retries=retries.get(s.tool, 0), retry_rate=s.retry_rate,
                    p50_ms=s.p50_ms, p95_ms=s.p95_ms,
                    last_error_ts=last_error[s.tool].ts if s.tool in last_error else None,
                    last_error_type=last_error[s.tool].outcome if s.tool in last_error else None)
            for s in summarize_tools(kept)]
    return rows, info


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


SOURCE_ORDER = ("app", "eval", "script")


def _configured_model(tier: str, settings: Settings) -> str:
    return {"main": settings.models.main.id, "light": settings.models.light.id,
            "embedding": settings.models.embedding.id}.get(tier, "-")


def _ledger_view(settings: Settings, ledger: Path, tracker: Path) -> tuple[list[TierRow], float, float, list[SourceRow]]:
    rows = {tier: TierRow(tier=tier, model=_configured_model(tier, settings)) for tier in TIER_ORDER}
    sources: dict[str, SourceRow] = {}
    today = datetime.now(timezone.utc).date().isoformat()  # the ledger counts days in UTC, as the billing page does
    total = today_usd = 0.0
    for event in read_events(ledger):
        if event.get("event") != "spend":
            continue
        model = event.get("model")
        row = rows[_tier_of(model, settings)]
        row.ledger_calls += event.get("calls", 0)
        row.ledger_usd += event.get("usd", 0.0)
        if model and model != row.model and model not in row.seen_as:
            row.seen_as.append(model)  # a dated name, or for "other" the model missing from config
        total += event.get("usd", 0.0)
        if str(event.get("ts", ""))[:10] == today:
            today_usd += event.get("usd", 0.0)
            source = sources.setdefault(str(event.get("source", "unknown")), SourceRow(source=str(event.get("source", "unknown"))))
            source.usd += event.get("usd", 0.0)
            source.calls += event.get("calls", 0)
    for entry in _tracker_entries(tracker):
        if entry.step == "tool":
            continue  # tool steps have no tier and no price
        row = rows[entry.tier if entry.step == "llm" and entry.tier in rows else
                   "embedding" if entry.step == "embedding" else "other"]
        row.tracker_steps += 1
        row.tracker_usd += entry.usd
    for row in rows.values():
        row.ledger_usd, row.tracker_usd = round(row.ledger_usd, 6), round(row.tracker_usd, 6)
        row.seen_as.sort()
    ordered = sorted(sources.values(), key=lambda s: (SOURCE_ORDER.index(s.source) if s.source in SOURCE_ORDER
                                                      else len(SOURCE_ORDER), s.source))
    for source in ordered:
        source.usd = round(source.usd, 6)
    return list(rows.values()), round(total, 6), round(today_usd, 6), ordered


def session_cost(tracker: Any, thread_ids: Sequence[str]) -> SessionCost:
    """The request cost tracker's total for this browser session's conversations (an estimate: the ledger is the audit
    trail, but it does not know sessions)."""
    if tracker is None or not thread_ids:
        return SessionCost()
    mine = [e for e in tracker.entries if e.session_id in set(thread_ids)]
    return SessionCost(usd=round(sum(e.usd for e in mine), 6), llm_calls=sum(e.step == "llm" for e in mine),
                       tokens=sum(e.input_tokens + e.output_tokens for e in mine))


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
    results_dir: Path, ledger: Path | None = None, window: ToolWindowName = "all",
    hitl_window: HitlWindowName = "all", sessions: Sequence[str] = (),
    now: datetime | None = None, app_started: datetime | None = None,
) -> MonitoringData:
    data_dir = Path(data_dir)
    now = now or datetime.now(timezone.utc)
    tools, tool_window = _tool_view(data_dir, window, now, app_started or APP_STARTED)
    hitl_filter = {"session": {"sessions": list(sessions)}, "today": {"since": _day_start(now)}, "all": {}}[hitl_window]
    live = settings.modes.llm == "live"
    index_dir = data_dir / ("index" if live else "index_mock")  # the folders `build_stack` uses
    embedder = settings.models.embedding.id if live else embedder_id(HashingEmbeddings())
    tiers, ledger_total, ledger_today, sources = _ledger_view(settings, ledger or ledger_path(), data_dir / COST_FILE)
    return MonitoringData(
        tools=tools, tool_window=tool_window,
        components=[ComponentRow(component=h.component.value, status=h.status.value, running_instead=h.mode or "",
                                 detail=mask_emails_in(h.reason or ""), load_seconds=h.load_seconds)
                    for h in registry.all()],
        degraded=[DegradedRow(component=h.component.value, mode=h.mode or "", reason=mask_emails_in(h.reason or ""),
                              since=h.since.isoformat(timespec="seconds") if h.since else None)
                  for h in registry.degraded_modes()],
        tiers=tiers, ledger_total_usd=ledger_total, ledger_today_usd=ledger_today,
        sources_today=sources,
        budget=_budget_row(settings, data_dir / OPS_DB_FILE),
        stale=check_staleness(kb_dir, index_dir, settings.chunking, embedder),
        hitl=load_counts(data_dir / HITL_FILE, trace_path=data_dir / TRACE_FILE, **hitl_filter),
        hitl_label={"session": "This session", "today": "Today (UTC)", "all": "All time"}[hitl_window],
        drift=DriftHistory(data_dir / DRIFT_LOG).points(),
        agent_metrics=_agent_metrics(Path(results_dir)),
    )


# --- rendering ------------------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _choose(label: str, options: dict[str, Any], key: str) -> Any:
    return options[st.radio(label, list(options), index=len(options) - 1, horizontal=True, key=key)]


def _render_tools(data: MonitoringData) -> None:
    st.subheader("Tool stats")
    info = data.tool_window
    if not data.tools:
        st.caption(f"No tool calls since {info.since}." if info.since else "No tool calls yet.")
    else:
        st.caption(f"{info.label}: {info.calls} calls since {info.since}, the latest at {info.last_ts}.")
        st.dataframe(pd.DataFrame([{
            "tool": r.tool, "calls": r.calls, "success rate": round(r.success_rate, 3), "failures": r.failures,
            "failed by": ", ".join(f"{k}: {v}" for k, v in sorted(r.failure_types.items())) or "-",
            "last error": f"{r.last_error_ts} ({r.last_error_type})" if r.last_error_ts else "-",
            "retries": r.retries, "retry rate": round(r.retry_rate, 3), "p50 ms": round(r.p50_ms, 1),
            "p95 ms": round(r.p95_ms, 1)} for r in data.tools]), hide_index=True, width="stretch")
    if info.undated:
        st.caption(f"{info.undated} older call(s) were logged before times were recorded and cannot be dated; "
                   "they appear only under All time.")


def _render_degraded(data: MonitoringData) -> None:
    st.subheader("Components and degraded modes")
    st.dataframe(pd.DataFrame([{
        "component": c.component, "status": "not checked yet" if c.status == "unknown" else c.status,
        "load time (s)": None if c.load_seconds is None else round(c.load_seconds, 2),
        "running instead": c.running_instead or "-", "detail": c.detail or "-"} for c in data.components]),
        hide_index=True, width="stretch")
    st.caption("Skipped means left out on purpose (the reranker in mock mode); it is not a fault. "
               "Degraded means a component failed or is not configured, and the substitute shown is running.")


def _render_cost(ctx: UiContext, data: MonitoringData) -> None:
    st.subheader("Cost by tier")
    left, middle, right = st.columns(3)
    left.metric("Today (UTC), all sources", f"${data.ledger_today_usd:.4f}",
                help="Everything the spend ledger recorded today: this app, eval runs, scripts and drift checks.")
    session = session_cost(ctx.service.services.cost, st.session_state.get("thread_ids", []))
    middle.metric("This session (tracker estimate)", f"${session.usd:.4f}",
                  help=f"{session.llm_calls} model calls and {session.tokens:,} tokens in this browser session's "
                       "conversations, from the request cost tracker. Mock runs price nothing, so it stays at $0.")
    spent = data.budget.spent_today_usd
    right.metric("Daily budget used", "no spend recorded" if spent is None else f"${spent:.4f}",
                 help=f"Cutoff 1 at ${data.budget.cutoff_1_usd:.2f} (Light-Model only), "
                      f"cutoff 2 at ${data.budget.cutoff_2_usd:.2f} (service stops). Level now: {data.budget.level}.")
    st.caption(f"Ledger, all time: ${data.ledger_total_usd:.4f}.")
    if data.sources_today:
        st.dataframe(pd.DataFrame([{"today, by source": s.source, "ledger USD": s.usd, "ledger calls": s.calls}
                                   for s in data.sources_today]), hide_index=True, width="stretch")
    else:
        st.caption("Nothing in the ledger today.")
    st.dataframe(pd.DataFrame([{
        "tier": t.tier, "model": t.model, "also billed as": ", ".join(t.seen_as) or "-",
        "ledger calls (all time)": t.ledger_calls, "ledger USD (all time)": t.ledger_usd,
        "tracker steps": t.tracker_steps, "tracker USD": t.tracker_usd} for t in data.tiers]),
        hide_index=True, width="stretch")
    st.caption("The ledger is the audit trail to compare with the provider's billing page; the tracker is the "
               "per-request analytics. Dated model names are grouped under the model configured for their tier. "
               "'other' is a model missing from config.yaml (priced as the Main model).")


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
                         width="stretch")
        if stale.new_files:
            st.info(f"Not yet ingested: {', '.join(stale.new_files)}.")
    else:
        st.caption("No stale chunks: every indexed file matches the hash stored at ingest.")


def _render_hitl(data: MonitoringData) -> None:
    st.subheader("Human checks at the ticket approval gate")
    counts = data.hitl
    st.caption(f"{data.hitl_label}.")
    cells = [("Approvals", counts.approvals), ("Rejections", counts.rejections), ("Edits", counts.edits)]
    for column, (label, value) in zip(st.columns(3), cells):
        column.metric(label, str(value))
    split = [("Approved, ticket created", counts.approved_created), ("Approved, creation failed", counts.approved_failed)]
    if counts.approved_unknown:
        split.append(("Approved, outcome unknown", counts.approved_unknown))
    for column, (label, value) in zip(st.columns(len(split)), split):
        column.metric(label, str(value))
    st.caption("Counted when an employee answers the approval gate: yes, no, or a change to the draft. "
               "Unclear replies are not decisions, so they are not counted. Each approval is matched to what happened "
               "to the ticket afterwards; for older approvals that comes from the trace log, and \"unknown\" means "
               "neither log shows it. The log keeps everything; the filter only chooses what is counted.")


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


RATE_COLUMNS = {"faithfulness": "Faithfulness", "recall_at_k": "Recall@k", "escalation_rate": "Escalation rate"}


def _drift_frame(points: list[DriftPoint]) -> pd.DataFrame:
    index = pd.to_datetime([p.ts for p in points])
    return pd.DataFrame({label: [getattr(p, field) for p in points] for field, label in RATE_COLUMNS.items()}
                        | {"Average rerank score": [p.avg_rerank_score for p in points]}, index=index)


def _render_drift_history(ctx: UiContext, data: MonitoringData) -> None:
    mode = ctx.settings.modes.llm
    points = [p for p in data.drift if p.mode == mode]
    other = {m: sum(p.mode == m for p in data.drift) for m in ("mock", "live") if m != mode}
    if not points:
        st.caption(f"No drift checks yet in {mode} mode." if not data.drift else
                   f"No drift checks yet in {mode} mode ({other[next(iter(other))]} from the other mode are hidden).")
        return
    frame = _drift_frame(points)
    hidden = sum(other.values())
    st.caption(f"{len(points)} point{'s' if len(points) != 1 else ''} from {mode} runs"
               + (f"; {hidden} from {next(iter(other))} runs are hidden, since the two are not comparable" if hidden else "")
               + ". A line appears from the second check on.")
    st.line_chart(frame[list(RATE_COLUMNS.values())])
    st.line_chart(frame[["Average rerank score"]])
    with st.expander("Run details"):
        st.dataframe(pd.DataFrame([{
            "when": p.ts, "mode": p.mode, "faithfulness": p.faithfulness, f"recall@{p.k}": p.recall_at_k,
            "escalation": round(p.escalation_rate, 3), "avg rerank score": p.avg_rerank_score,
            "retrieval": f"{p.retrieval_strategy} (reranker {p.reranker_status})",
            "prompts": ", ".join(f"{name}_v{version}" for name, version in sorted(p.prompt_versions.items())),
            "models": ", ".join(f"{tier}: {model}" for tier, model in p.models.items()), "USD": p.usd}
            for p in reversed(points)]), hide_index=True, width="stretch")


def _drift_controls(ctx: UiContext) -> None:
    """The button flow: estimate first, run only after an explicit confirmation (FR-H15)."""
    state = st.session_state
    if notice := state.pop("drift_notice", None):
        (st.error if notice[0] == "error" else st.success)(notice[1])
    if not state.get("drift_pending"):
        if st.button("Check for drift", key="drift_start"):
            state["drift_pending"] = True
            st.rerun()
        return
    golden = Golden.load()
    for line in describe_estimate(estimate_drift(ctx.settings, golden.size), golden):
        st.write(line)
    confirm, cancel = st.columns(2)
    if cancel.button("Cancel", key="drift_cancel"):
        state["drift_pending"] = False
        st.rerun()
    if confirm.button("Confirm and run", key="drift_confirm", type="primary"):
        runner: DriftRunner = state.get("drift_runner") or make_runner(ctx.settings, golden=golden)
        state["drift_pending"] = False
        try:
            with st.spinner("Running the golden set"):
                point = runner()
            DriftHistory(ctx.data_dir / DRIFT_LOG).append(point)
            state["drift_notice"] = ("ok", "Drift check stored.")
        except DriftCeilingExceeded as stop:  # our own message: numbers only
            state["drift_notice"] = ("error", f"The drift check stopped: {stop}. Nothing was stored.")
        except Exception as error:  # the type only: a message could echo a prompt or an address (FR-G11)
            state["drift_notice"] = ("error", f"The drift check did not finish ({type(error).__name__}). "
                                              "Nothing was stored.")
        st.rerun()


def _render_drift(ctx: UiContext, data: MonitoringData) -> None:
    st.subheader("Drift")
    st.caption("Re-runs the golden set (KB set plus gap set) and charts Faithfulness (the runtime grounding judge's "
               "accept rate), Recall@k, escalation rate and the average top rerank score over time.")
    _drift_controls(ctx)
    _render_drift_history(ctx, data)


def render_monitoring(ctx: UiContext) -> None:
    if st.button("Refresh", key="monitoring_refresh"):
        st.rerun()
    tool_window = _choose("Tool stats window", TOOL_WINDOWS, "monitoring_tool_window")
    hitl_window = _choose("Human checks window", HITL_WINDOWS, "monitoring_hitl_window")
    data = load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir,
                           window=tool_window, hitl_window=hitl_window,
                           sessions=st.session_state.get("thread_ids", []))
    _render_tools(data)
    _render_degraded(data)
    _render_cost(ctx, data)
    _render_stale(data)
    _render_hitl(data)
    _render_agent_metrics(data)
    _render_drift(ctx, data)

#!/usr/bin/env python
"""Throughput of the whole chat pipeline in mock mode (T9.5, FR-H11): requests per second and p50/p95 latency.

    python scripts/load_test.py [--requests 200] [--workers 8] [--out evals/results/load_test.json]

The real service (guards, orchestrator, workers, output guard) runs in this process with mock models and mock tickets,
a temp data folder and a temp spend ledger, so it never touches `data/`, the network or any paid service. Mock models
answer instantly, so the numbers measure our own code (retrieval, rerank, guards, graph, SQLite), not OpenAI's latency;
a live run is bounded by the model calls and will be much slower. Each request is a fresh session, which also keeps the
per-session rate limits (bypassed here) out of the measurement.
"""
import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = (
    "How do I reset my forgotten password?",
    "How many casual leave days do I get?",
    "How do I reset my password, and how many casual leave days do I get?",
    "How do I register my tractor for the office parking lot?",  # a gap: the mock escalates it
    "Can you check the status of my tickets? My email is one.ticket@smartdesk.example",
    "What is the server name for the VPN app?",
)


def percentile(sorted_values: list[float], share: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    index = max(0, min(len(sorted_values) - 1, math.ceil(share * len(sorted_values)) - 1))
    return sorted_values[index]


def run_load(requests: int, workers: int) -> dict:
    """Send `requests` messages from `workers` threads; return the summary. Temp data, mock modes, no network."""
    from smartdesk_ai.config.settings import DEFAULT_CONFIG
    from smartdesk_ai.ops.health import HealthRegistry
    from smartdesk_ai.ui.context import build_ui_context

    with tempfile.TemporaryDirectory(prefix="smartdesk-load-") as tmp:
        work = Path(tmp)
        config = work / "config.yaml"
        config.write_text(DEFAULT_CONFIG.read_text().replace("bypass: false", "bypass: true", 1))
        environ = {
            "SMARTDESK_LLM_MODE": "mock", "SMARTDESK_TICKETS_MODE": "mock",
            "SMARTDESK_DATA_DIR": str(work / "data"), "SMARTDESK_LEDGER_PATH": str(work / "ledger.jsonl"),
        }
        os.environ["SMARTDESK_LEDGER_PATH"] = environ["SMARTDESK_LEDGER_PATH"]
        context = build_ui_context(environ=environ, dotenv_path=None, config_path=config, registry=HealthRegistry())
        service = context.service
        service.handle("warm-up", QUESTIONS[0])  # loads the index and the reranker outside the timed window

        def one(number: int) -> tuple[float, bool]:
            started = time.perf_counter()
            try:
                reply = service.handle(f"load-{number}", QUESTIONS[number % len(QUESTIONS)])
                ok = bool(reply.text)
            except Exception:  # a failed request is counted, never raised
                ok = False
            return time.perf_counter() - started, ok

        wall = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, range(requests)))
        wall = time.perf_counter() - wall

    latencies = sorted(seconds for seconds, _ in results)
    return {
        "mode": "mock", "requests": requests, "workers": workers,
        "failed": sum(1 for _, ok in results if not ok),
        "wall_seconds": round(wall, 3),
        "requests_per_second": round(requests / wall, 2),
        "latency_p50_seconds": round(percentile(latencies, 0.50), 4),
        "latency_p95_seconds": round(percentile(latencies, 0.95), 4),
        "latency_mean_seconds": round(statistics.fmean(latencies), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mock-mode throughput of the chat pipeline")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None, help="also write the summary here as JSON")
    args = parser.parse_args(argv)
    summary = run_load(args.requests, args.workers)
    print(f"{summary['requests']} requests, {summary['workers']} workers (mock models, mock tickets)")
    print(f"throughput: {summary['requests_per_second']} requests/s over {summary['wall_seconds']} s")
    print(f"latency: p50 {summary['latency_p50_seconds']} s, p95 {summary['latency_p95_seconds']} s, "
          f"mean {summary['latency_mean_seconds']} s; failed: {summary['failed']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI chat: python -m smartdesk_ai.agent [--debug]."""
import argparse
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from smartdesk_ai.agent.chat_service import Responder, build_service, stub_responder
from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
from smartdesk_ai.ops.health import HealthRegistry, get_registry

EXIT_WORDS = {"quit", "exit"}


def format_debug(debug: dict[str, Any]) -> list[str]:
    """Debug lines for one reply. Retrieved chunks are ranked with their scores, then the answer."""
    lines = []
    if "chunks" in debug:
        chunks = debug["chunks"]
        if not chunks:
            lines.append("[debug] no chunks retrieved")
        for rank, chunk in enumerate(chunks, 1):
            lines.append(f"[debug] {rank}. {chunk['entry_id']} score={chunk['score']:.3f} {chunk['title']}")
    if "answer" in debug:
        lines.append(f"[debug] answer: {debug['answer']}")
    lines.extend(f"[debug] {key}={value}" for key, value in debug.items() if key not in ("chunks", "answer"))
    return lines


def main(
    argv: Sequence[str] | None = None,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    registry: HealthRegistry | None = None,
    responder: Responder | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m smartdesk_ai.agent", description="SmartDesk AI CLI chat")
    parser.add_argument("--debug", action="store_true", help="show debug details for each reply")
    args = parser.parse_args(argv)

    registry = registry if registry is not None else get_registry()
    service = build_service(
        environ=environ, dotenv_path=dotenv_path, registry=registry, responder=responder or stub_responder
    )

    output("SmartDesk AI. Type 'quit' to leave.")
    missing = load_env(environ=environ, dotenv_path=dotenv_path).missing_required()
    if missing:
        output(f"[notice] not set: {', '.join(missing)} (mock mode only)")
    for health in registry.degraded_modes():
        output(f"[degraded] {health.component.value}: {health.mode} ({health.reason})")

    session_id = uuid.uuid4().hex
    while True:
        try:
            line = input_fn("> ")
        except (EOFError, KeyboardInterrupt):
            break
        line = line.strip()
        if line.lower() in EXIT_WORDS:
            break
        if not line:
            continue
        reply = service.handle(session_id, line)
        output(reply.text)
        if args.debug:
            for debug_line in format_debug(reply.debug):
                output(debug_line)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())

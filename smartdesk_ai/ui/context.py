"""What the Streamlit app needs besides the chat service: where the runtime files are, the settings, the
health registry and the KB folder. `build_ui_context` is the same assembly as the CLI (`agent/__main__.py`), so
both front ends run the same graph. Live mode that cannot start raises `LiveModeUnavailable`; the app shows it."""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from smartdesk_ai.agent.chat_service import ChatService, build_service
from smartdesk_ai.agent.factory import build_responder, resolve_data_dir
from smartdesk_ai.config.env import DEFAULT_DOTENV
from smartdesk_ai.config.settings import DEFAULT_CONFIG, Settings
from smartdesk_ai.ops.health import HealthRegistry, get_registry
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ROOT
from smartdesk_ai.tickets import TicketClient

DEFAULT_RESULTS_DIR = ROOT / "evals" / "results"  # where `python -m evals.run --out-dir` writes (agent metrics)


@dataclass(frozen=True)
class UiContext:
    service: ChatService
    settings: Settings
    data_dir: Path
    registry: HealthRegistry
    kb_dir: Path
    results_dir: Path = DEFAULT_RESULTS_DIR


def build_ui_context(
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    data_dir: Path | None = None,
    registry: HealthRegistry | None = None,
    config_path: Path = DEFAULT_CONFIG,
    kb_dir: Path = DEFAULT_KB_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    client: TicketClient | None = None,
) -> UiContext:
    registry = registry if registry is not None else get_registry()
    resolved = resolve_data_dir(data_dir, environ)
    service = build_service(
        environ=environ, dotenv_path=dotenv_path, registry=registry, config_path=config_path, kb_dir=kb_dir,
        responder_factory=lambda settings: build_responder(
            settings, environ=environ, dotenv_path=dotenv_path, data_dir=resolved, kb_dir=kb_dir, registry=registry,
            client=client),
    )
    return UiContext(service=service, settings=service.settings, data_dir=resolved, registry=registry, kb_dir=kb_dir,
                     results_dir=results_dir)

"""Builds the UI's context in mock mode inside a temp folder, and an AppTest around the app script."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ui.context import UiContext, build_ui_context

APP = Path(__file__).resolve().parents[1] / "smartdesk_ai" / "ui" / "app.py"


def mock_context(tmp_path: Path, **kwargs: object) -> UiContext:
    """Mock models, mock tickets, no .env, every runtime file under tmp_path (never the repo's data/)."""
    kwargs.setdefault("registry", HealthRegistry())
    kwargs.setdefault("results_dir", tmp_path / "results")
    return build_ui_context(environ={}, dotenv_path=None, data_dir=tmp_path, **kwargs)


PAGES = {"chat": "views/chat_page.py", "monitoring": "views/monitoring_page.py"}  # relative to app.py


def app_test(ctx: UiContext, *, page: str = "chat", **session: object) -> AppTest:
    """The app on one of its two pages. `at.main` is that page's body."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["ctx"] = ctx
    for key, value in session.items():
        at.session_state[key] = value
    if page != "chat":
        at.switch_page(PAGES[page])
    return at.run()


def monitoring_app(ctx: UiContext, **session: object) -> AppTest:
    return app_test(ctx, page="monitoring", **session)


def element_text(node) -> str:
    """Everything a rendered block shows as text: markdown, captions, metrics and the cells of data frames."""
    parts: list[str] = []
    for kind in ("markdown", "caption", "text", "metric", "subheader", "header", "warning", "info", "error", "success"):
        parts.extend(str(getattr(e, "value", e)) for e in getattr(node, kind, []))
    parts.extend(frame.value.to_csv() for frame in node.dataframe)
    return "\n".join(parts)


def charts(node) -> list:
    """The line charts of a rendered block, each as a frame with one column per series and one row per point."""
    from streamlit import dataframe_util

    frames = []
    for chart in node.get("vega_lite_chart"):
        long = dataframe_util.convert_arrow_bytes_to_pandas_df(chart.proto.datasets[0].data.data)
        long = long.rename(columns=lambda name: name.split(" -- ")[0])  # Streamlit's "index -- streamlit-generated"
        if "color" in long.columns:  # several series come as (index, color = series name, value)
            long = long.pivot(index="index", columns="color", values="value")
        else:  # one series is already wide
            long = long.set_index("index")
        long.columns.name = None
        frames.append(long)
    return frames

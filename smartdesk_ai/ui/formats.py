"""One format per kind of number, so the two pages read the same."""


def usd(value: float) -> str:
    return f"${value:.4f}"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"

"""Import Ragas (pinned 0.4.3) in this environment.

`ragas/llms/base.py` does `from langchain_community.chat_models.vertexai import ChatVertexAI` at import
time, a module that langchain-community 0.4.2 (pinned for LangChain 1.x) no longer ships, so a plain
`import ragas` fails with ModuleNotFoundError. We never use Vertex AI, so a placeholder module with an empty
`ChatVertexAI` class is registered first. Remove this shim when Ragas or langchain-community is re-pinned
so that the plain import works. The legacy `ragas.metrics` classes are used because the pinned version's
new `ragas.metrics.collections` API needs an instructor client (see evals/README.md).
"""
import sys
import types
import warnings

_MISSING = "langchain_community.chat_models.vertexai"


def import_ragas():
    """The imported `ragas` package, with the shim applied only if the plain import fails."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import ragas
        return ragas
    except ModuleNotFoundError as error:
        if _MISSING not in str(error):
            raise
    for name in [m for m in sys.modules if m == "ragas" or m.startswith("ragas.")]:
        del sys.modules[name]  # drop the half-imported package
    stub = types.ModuleType(_MISSING)
    stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
    sys.modules[_MISSING] = stub
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import ragas
    return ragas

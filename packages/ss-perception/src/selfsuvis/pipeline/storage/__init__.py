"""Perception storage helpers (vector index)."""

from importlib import import_module
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_EXPORTS = {
    "InMemoryStore": (".vector_store", "InMemoryStore"),
    "QdrantStore": (".qdrant", "QdrantStore"),
    "RecentEmbeddingIndex": (".recent_index", "RecentEmbeddingIndex"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr_name)

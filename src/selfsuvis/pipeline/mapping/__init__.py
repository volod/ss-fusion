"""Mapping, geometry, and splat helpers."""

from importlib import import_module

_EXPORTS = {
    "advise_map_quality": (".quality_advisor", "advise_map_quality"),
    "build_sparse_map": (".builder", "build_sparse_map"),
    "build_semantic_environment_graph": (".semantic_graph", "build_semantic_environment_graph"),
    "export_ply": (".builder", "export_ply"),
    "open_3d_viewers": (".viewer", "open_3d_viewers"),
    "run_sfm": (".sfm", "run_sfm"),
    "view_npz": (".viewer", "view_npz"),
    "write_semantic_graph_markdown": (".semantic_graph", "write_semantic_graph_markdown"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr_name)

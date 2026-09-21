"""python: and http: step bindings."""

import importlib
from collections.abc import Callable
from typing import Any


def parse_binding(value: str) -> tuple[str, str]:
    if value.startswith("python:"):
        target = value[len("python:") :]
        if ":" not in target:
            raise ValueError(f"python binding needs module:function: {value}")
        return "python", target
    if value.startswith("http://") or value.startswith("https://"):
        return "http", value
    if value.startswith("http:"):
        return "http", value[len("http:") :]
    raise ValueError(f"unsupported binding: {value}")


def load_python(target: str) -> Callable[..., Any]:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    func = getattr(module, attr)
    if not callable(func):
        raise TypeError(f"{target} is not callable")
    return func


def run_python(target: str, context: dict[str, Any]) -> dict[str, Any]:
    result = load_python(target)(context)
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise TypeError(f"{target} must return a dict")
    return result


def run_http(target: str, context: dict[str, Any]) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("httpx is required for http: bindings") from exc
    payload = {
        "step_id": context.get("step_id"),
        "parameters": context.get("parameters") or {},
        "ports": context.get("ports") or {},
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(target, json=payload)
        response.raise_for_status()
        data = response.json()
    if isinstance(data, dict):
        return data
    return {"result": data}

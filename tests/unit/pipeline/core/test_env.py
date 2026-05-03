import os
from pathlib import Path

from selfsuvis.pipeline.core.env import env_csv, env_json_dict, set_env_if_present


def test_env_csv_parses_trimmed_values(monkeypatch):
    monkeypatch.setenv("TEST_CSV", " alpha, beta ,gamma ")
    assert env_csv("TEST_CSV") == ["alpha", "beta", "gamma"]


def test_env_json_dict_uses_fallback_and_error_callback(monkeypatch):
    seen: list[str] = []

    def _on_error(message: str, key: str) -> None:
        seen.append(message % key)

    monkeypatch.setenv("TEST_JSON", "{bad")
    parsed = env_json_dict("TEST_JSON", default={"fallback": "yes"}, on_error=_on_error)

    assert parsed == {"fallback": "yes"}
    assert seen == ["TEST_JSON contains invalid JSON; using default value"]


def test_set_env_if_present_does_not_write_empty(monkeypatch):
    monkeypatch.delenv("MAYBE_SET", raising=False)
    set_env_if_present("MAYBE_SET", "")
    assert "MAYBE_SET" not in os.environ

    set_env_if_present("MAYBE_SET", Path("/tmp/value"))
    assert os.environ["MAYBE_SET"] == "/tmp/value"

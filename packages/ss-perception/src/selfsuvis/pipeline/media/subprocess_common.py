"""Shared subprocess helpers for media pipelines."""

import subprocess


def run_checked(cmd: list[str], *, timeout: int) -> None:
    subprocess.run(cmd, check=True, timeout=timeout)


def run_captured(
    cmd: list[str],
    *,
    timeout: int,
    text: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=text,
        timeout=timeout,
    )

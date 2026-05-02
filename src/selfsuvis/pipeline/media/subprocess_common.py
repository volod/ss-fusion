"""Shared subprocess helpers for media pipelines."""

import subprocess
from typing import List


def run_checked(cmd: List[str], *, timeout: int) -> None:
    subprocess.run(cmd, check=True, timeout=timeout)


def run_captured(
    cmd: List[str],
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

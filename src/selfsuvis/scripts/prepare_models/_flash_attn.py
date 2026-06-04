"""flash-attn installation helper."""

import sys
import time

from selfsuvis.pipeline.core.logging import get_logger

log = get_logger("prepare_models")


def _install_flash_attn() -> None:
    """Install flash-attn using prebuilt PyPI wheel or compile from source.

    Uses ``--no-build-isolation`` so nvcc and torch headers from the active
    environment are used for source builds (avoids version mismatches).
    Prebuilt wheels exist on PyPI for the most common CUDA + Python + torch
    combinations and are used automatically when available.
    """
    log.info("flash-attn — checking installation …")
    try:
        import flash_attn

        log.info("  [ok] flash-attn already installed  (version %s)", flash_attn.__version__)
        return
    except ImportError:
        pass

    import torch

    if not torch.cuda.is_available():
        log.warning(
            "  CUDA not available — skipping flash-attn installation.\n"
            "  flash-attn is a CUDA-only package and cannot run on CPU."
        )
        return

    log.info(
        "  Installing flash-attn (uses prebuilt wheel when available; "
        "otherwise compiles from source — may take several minutes) …"
    )
    import math as _math
    import subprocess as _sp

    def _flash_attn_max_jobs(ram_per_job_gb: float = 12.0, reserve_frac: float = 0.20) -> int:
        try:
            meminfo = {}
            for line in open("/proc/meminfo").readlines():
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", 0)
        except Exception:
            total_kb = avail_kb = 0
        total_gb = total_kb / 1024 / 1024
        avail_gb = avail_kb / 1024 / 1024
        floor_gb = total_gb * reserve_frac
        usable_gb = max(0.0, avail_gb - floor_gb)
        raw_mem = usable_gb / ram_per_job_gb if ram_per_job_gb > 0 else 0.0
        mem_jobs = _math.ceil(raw_mem) if (raw_mem % 1) >= 0.8 else int(raw_mem)
        try:
            import os as _os
            cpu_jobs = max(1, (_os.cpu_count() or 4) // 2)
        except Exception:
            cpu_jobs = 1
        jobs = max(1, min(mem_jobs, cpu_jobs))
        log.info(
            "  flash-attn compilation budget: %.1f GiB total / %.1f GiB avail / %.1f GiB usable"
            " → mem_jobs=%d  cpu_jobs=%d  MAX_JOBS=%d",
            total_gb, avail_gb, usable_gb, mem_jobs, cpu_jobs, jobs,
        )
        return jobs

    import os as _os
    ram_per_job = float(_os.environ.get("FLASH_ATTN_RAM_PER_JOB_GB", "12"))
    max_jobs = _flash_attn_max_jobs(ram_per_job_gb=ram_per_job)

    _sp.run([sys.executable, "-m", "pip", "install", "wheel", "packaging", "-q"], check=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "flash-attn",
        "--no-build-isolation",
        "-q",
    ]
    t0 = time.monotonic()
    result = _sp.run(cmd, env={**_os.environ, "MAX_JOBS": str(max_jobs)}, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "flash-attn installation failed.\n"
            "  Ensure CUDA toolkit is installed (nvcc must be in PATH).\n"
            "  Prebuilt wheels for your CUDA + Python + torch combination:\n"
            "    https://github.com/Dao-AILab/flash-attention/releases\n"
            "  Download the matching .whl and install manually:\n"
            "    pip install <wheel_file.whl>"
        )
    log.info("  [ok] flash-attn installed  (%.1fs)", time.monotonic() - t0)

"""HuggingFace auth-error detection and interactive retry logic."""

import os
import sys

from selfsuvis.pipeline.core.logging import get_logger

from ._ollama import _UNIDRIVE_COLLECTION_URL

log = get_logger("prepare_models")


def _is_auth_error(exc: Exception):
    """Return (is_auth_error: bool, kind: str) where kind includes repo-not-found."""
    for module_path in ("huggingface_hub.errors", "huggingface_hub.utils"):
        try:
            m = __import__(module_path, fromlist=["GatedRepoError", "RepositoryNotFoundError"])
            GatedRepoError = getattr(m, "GatedRepoError", None)
            if GatedRepoError and isinstance(exc, GatedRepoError):
                return True, "gated"
            RepositoryNotFoundError = getattr(m, "RepositoryNotFoundError", None)
            if RepositoryNotFoundError and isinstance(exc, RepositoryNotFoundError):
                return False, "repo_not_found"
        except (ImportError, AttributeError):
            continue

    msg = str(exc).lower()
    if "gated" in msg or ("access" in msg and "repo" in msg):
        return True, "gated"
    if "repository not found" in msg or "404" in msg:
        return False, "repo_not_found"
    if "401" in msg or "403" in msg or "unauthorized" in msg or "authentication" in msg:
        return True, "unauthorized"
    return False, ""


def _with_auth_retry(label: str, model_id: str, download_fn) -> None:
    """Run download_fn(); on auth/gated error switch to interactive mode.

    Prints step-by-step HuggingFace authentication instructions and waits for
    the user to complete them, then retries automatically.  Gives up after 3
    failed attempts.  In non-interactive (piped) mode prints instructions and
    raises so the caller can log the error.
    """
    max_retries = 3
    attempts = 0
    while True:
        attempts += 1
        try:
            download_fn()
            return
        except Exception as exc:
            is_auth, kind = _is_auth_error(exc)
            if kind == "repo_not_found":
                raise RuntimeError(
                    f"Hugging Face repo '{model_id}' was not found. "
                    f"UniDriveVLA weights are currently published under the owl10 collection: "
                    f"{_UNIDRIVE_COLLECTION_URL}"
                ) from exc
            if not is_auth or attempts >= max_retries:
                raise

            hf_url = f"https://huggingface.co/{model_id}"
            bar = "-" * 70
            print(f"\n{bar}", flush=True)
            print(f"  ACCESS REQUIRED — {label}", flush=True)
            print(f"{bar}", flush=True)

            if kind == "gated":
                print(
                    f"""
  This model is gated.  You must accept its license on HuggingFace before
  the weights can be downloaded.

    1. Open the model page in your browser:
         {hf_url}
       Click  "Agree and access repository"

    2. Generate a HuggingFace token (if you don't have one):
         https://huggingface.co/settings/tokens
       Choose "Read" permissions.

    3. Enter your token at the prompt below (it will be set for this session),
       or leave blank if you already ran  huggingface-cli login.

    4. The download will retry automatically.
""",
                    flush=True,
                )
            else:
                print(
                    f"""
  This model requires a HuggingFace account / token.

    1. Create an account (if needed):
         https://huggingface.co/join

    2. Generate a Read token:
         https://huggingface.co/settings/tokens

    3. If the model page shows a license, accept it at:
         {hf_url}

    4. Enter your token at the prompt below (it will be set for this session),
       or leave blank if you already ran  huggingface-cli login.
""",
                    flush=True,
                )

            if not sys.stdin.isatty():
                print(
                    "  Running non-interactively — cannot prompt.\n"
                    "  Set HF_TOKEN env var and re-run this script in an interactive terminal.\n"
                    f"{bar}\n",
                    flush=True,
                )
                raise

            try:
                import getpass

                token_input = getpass.getpass(
                    "  HF token (leave blank to skip token entry, 's' to skip model): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(flush=True)
                raise exc

            if token_input.lower() == "s":
                log.warning("Skipping %s (user chose to skip)", label)
                raise exc

            if token_input and token_input.lower() != "s":
                os.environ["HF_TOKEN"] = token_input
                try:
                    from huggingface_hub import login as hf_login

                    hf_login(token=token_input, add_to_git_credential=False)
                    log.info("HuggingFace token accepted.")
                except Exception:
                    pass

            log.info("Retrying download: %s …", label)

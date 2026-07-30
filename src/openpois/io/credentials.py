#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Load Source Cooperative temporary S3 credentials.

Source Coop moved to an OIDC/STS model: credentials are minted by the
``source-coop`` CLI (browser login, then a short-lived token from the data
proxy) rather than issued as static keys in the dashboard. So the preferred
source is the CLI itself, which is self-refreshing as long as the cached login
is valid:

    source-coop login                 # once, opens a browser
    source-coop creds                 # credential_process JSON, what we read

:func:`load_source_coop_credentials` tries the CLI first and falls back to a
local ``.env.json``. The fallback file format is four keys:

    {
      "aws_access_key_id": "ASIA...",
      "aws_secret_access_key": "...",
      "aws_session_token": "...",
      "region_name": "us-west-2"
    }

If the file has not been touched recently we warn the caller (but do not fail);
STS tokens typically last an hour or so.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

DEFAULT_REGION = "us-west-2"

REQUIRED_KEYS = (
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "region_name",
)

REFRESH_HINT = (
    "Regenerate temporary credentials at "
    "https://source.coop/repositories/henryspatialanalysis/openpois/manage "
    "and write them to .env.json at the repo root."
)

STALE_SECONDS = 60 * 60  # Source Coop tokens usually last ~1 hour.

CLI_BINARY = "source-coop"
CLI_HINT = (
    "Install the Source Coop CLI and authenticate:\n"
    "  curl --proto '=https' --tlsv1.2 -LsSf https://github.com/"
    "source-cooperative/source-coop-cli/releases/latest/download/"
    "source-coop-cli-installer.sh | sh\n"
    "  source-coop login"
)


def _cli_path() -> str | None:
    """Locate the source-coop binary, including the cargo bin dir."""
    found = shutil.which(CLI_BINARY)
    if found:
        return found
    candidate = Path.home() / ".cargo" / "bin" / CLI_BINARY
    return str(candidate) if candidate.exists() else None


def load_credentials_from_cli(verbose: bool = True) -> dict | None:
    """Temporary credentials from ``source-coop creds``, or None if unavailable.

    Returns the same key names as the ``.env.json`` path so callers do not care
    which source was used. ``region_name`` is not part of the CLI payload; the
    proxy ignores it but boto3 requires one, so it is defaulted here.
    """
    binary = _cli_path()
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, "creds"], capture_output = True, text = True,
            timeout = 60, check = True,
        )
        payload = json.loads(result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, OSError):
        return None
    if not payload.get("AccessKeyId"):
        return None
    if verbose:
        expiry = payload.get("Expiration")
        print(f"Using Source Coop CLI credentials (expire {expiry}).")
    return {
        "aws_access_key_id": payload["AccessKeyId"],
        "aws_secret_access_key": payload["SecretAccessKey"],
        "aws_session_token": payload["SessionToken"],
        "region_name": DEFAULT_REGION,
        "expiration": payload.get("Expiration"),
    }


def load_source_coop_credentials(env_file: Path | str | None = None) -> dict:
    """Read Source Coop temporary S3 credentials.

    Prefers the ``source-coop`` CLI; falls back to ``env_file``, which defaults
    to ``~/repos/openpois/.env.json``.
    Raises ``FileNotFoundError`` or ``ValueError`` with a refresh hint if the
    file is missing or malformed. Prints a warning if the file's mtime is
    older than ~1 hour (tokens may have expired).
    """
    from_cli = load_credentials_from_cli()
    if from_cli is not None:
        return from_cli

    if env_file is None:
        env_file = Path.home() / "repos" / "openpois" / ".env.json"
    env_file = Path(env_file).expanduser()

    if not env_file.exists():
        raise FileNotFoundError(
            f"Source Coop credentials file not found at {env_file}. "
            f"No Source Coop CLI credentials either. {CLI_HINT}"
        )

    with env_file.open() as f:
        creds = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in creds]
    if missing:
        raise ValueError(
            f"Source Coop credentials file {env_file} is missing keys: "
            f"{missing}. {REFRESH_HINT}"
        )

    mtime_age = time.time() - env_file.stat().st_mtime
    if mtime_age > STALE_SECONDS:
        minutes = int(mtime_age // 60)
        print(
            f"⚠️  {env_file} was last updated ~{minutes} minutes ago. "
            "Source Coop tokens usually expire within an hour — if uploads "
            f"fail with ExpiredToken, {REFRESH_HINT}"
        )

    return {k: creds[k] for k in REQUIRED_KEYS}

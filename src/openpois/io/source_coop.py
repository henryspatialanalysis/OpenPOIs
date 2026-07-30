#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Source Cooperative upload helpers.

Source Coop serves its S3-compatible API through a data proxy at
``https://data.source.coop``, where the **bucket is the account** and the
repository is the first key segment: ``s3://{account}/{repo}/…``. Public reads
are at ``https://data.source.coop/{account}/{repo}/…``.

This replaced a direct-S3 layout that used the literal bucket
``us-west-2.opendata.source.coop`` with ``{account}/{repo}/…`` keys. That path
no longer authenticates: proxy-issued credentials are rejected by AWS S3 with
``InvalidAccessKeyId``, and the legacy bucket does not exist behind the proxy.
Both the ``endpoint_url`` and the account-as-bucket addressing are therefore
required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import boto3
from tqdm import tqdm

DEFAULT_BUCKET = "henryspatialanalysis"
DEFAULT_READ_HOST = "https://data.source.coop"
DEFAULT_ENDPOINT = "https://data.source.coop"


def make_client(creds: dict, endpoint_url: str = DEFAULT_ENDPOINT):
    """Return a boto3 S3 client pointed at the Source Coop data proxy.

    ``endpoint_url`` is not optional in practice: without it boto3 talks to AWS
    S3, which does not recognize proxy-issued credentials.
    """
    return boto3.client(
        "s3",
        aws_access_key_id = creds["aws_access_key_id"],
        aws_secret_access_key = creds["aws_secret_access_key"],
        aws_session_token = creds["aws_session_token"],
        region_name = creds.get("region_name") or "us-west-2",
        endpoint_url = endpoint_url,
    )


def upload_directory(
    client,
    local_dir: Path,
    bucket: str,
    key_prefix: str,
    patterns: Iterable[str] = ("*.parquet",),
    content_type: str = "application/octet-stream",
    dry_run: bool = False,
) -> list[str]:
    """Upload every file matching ``patterns`` under ``local_dir`` to the
    Source Coop bucket, preserving relative paths beneath ``key_prefix``.

    Returns the list of remote keys (whether or not ``dry_run`` was set).
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(local_dir.rglob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No files matching {list(patterns)} under {local_dir}."
        )

    key_prefix = key_prefix.rstrip("/")
    keys: list[str] = []
    iterator = tqdm(files, desc = f"→ {key_prefix}/", unit = "file")
    for local_path in iterator:
        relative = local_path.relative_to(local_dir).as_posix()
        key = f"{key_prefix}/{relative}"
        keys.append(key)
        if dry_run:
            continue
        client.upload_file(
            Filename = str(local_path),
            Bucket = bucket,
            Key = key,
            ExtraArgs = {
                "ACL": "bucket-owner-full-control",
                "ContentType": content_type,
            },
        )

    return keys


def upload_file(
    client,
    local_path: Path,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
    dry_run: bool = False,
) -> str:
    """Upload one local file to Source Coop; return the public HTTPS URL.

    Passes ``ACL=bucket-owner-full-control`` as required by Source Coop's
    upload policy.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    size_mb = local_path.stat().st_size / 1e6
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}→ s3://{bucket}/{key} ({size_mb:.1f} MB)")
    if not dry_run:
        client.upload_file(
            Filename = str(local_path),
            Bucket = bucket,
            Key = key,
            ExtraArgs = {
                "ACL": "bucket-owner-full-control",
                "ContentType": content_type,
            },
        )
    return f"{DEFAULT_READ_HOST}/{key}"


def upload_bytes(
    client,
    data: bytes,
    bucket: str,
    key: str,
    content_type: str = "text/markdown",
    dry_run: bool = False,
) -> str:
    """Upload an in-memory bytes payload (e.g. a generated README) to Source Coop."""
    size_kb = len(data) / 1e3
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}→ s3://{bucket}/{key} ({size_kb:.1f} KB, in-memory)")
    if not dry_run:
        client.put_object(
            Bucket = bucket,
            Key = key,
            Body = data,
            ACL = "bucket-owner-full-control",
            ContentType = content_type,
        )
    return f"{DEFAULT_READ_HOST}/{key}"


def public_url(key: str) -> str:
    """Compose the public read URL for a given object key."""
    return f"{DEFAULT_READ_HOST}/{key.lstrip('/')}"


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    """Return every object key in ``bucket`` under ``prefix`` (paginated)."""
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket = bucket, Prefix = prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])
    return keys


def delete_keys(
    client,
    bucket: str,
    keys: list[str],
    dry_run: bool = False,
) -> None:
    """Delete ``keys`` from ``bucket`` in batches of up to 1000 per request."""
    if not keys:
        return
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}deleting {len(keys)} object(s) from s3://{bucket}/")
    if dry_run:
        return
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        client.delete_objects(
            Bucket = bucket,
            Delete = {
                "Objects": [{"Key": k} for k in chunk],
                "Quiet": True,
            },
        )


def mirror_prefix(
    client,
    bucket: str,
    src_prefix: str,
    dst_prefix: str,
    dry_run: bool = False,
) -> dict:
    """Server-side copy every object under ``src_prefix`` to ``dst_prefix``.

    Any objects currently under ``dst_prefix`` whose relative path is not
    present in ``src_prefix`` are deleted first, so ``dst_prefix`` becomes a
    faithful mirror of ``src_prefix``. Returns a summary dict with the
    counts and the resolved src/dst prefixes.
    """
    src_prefix = src_prefix if src_prefix.endswith("/") else src_prefix + "/"
    dst_prefix = dst_prefix if dst_prefix.endswith("/") else dst_prefix + "/"

    # Without a live client we can't enumerate either side; report zeros.
    if client is None:
        src_keys: list[str] = []
        dst_keys: list[str] = []
    else:
        src_keys = list_keys(client, bucket, src_prefix)
        dst_keys = list_keys(client, bucket, dst_prefix)

    src_relatives = {k[len(src_prefix) :] for k in src_keys}
    stale = [k for k in dst_keys if k[len(dst_prefix) :] not in src_relatives]

    delete_keys(client, bucket, stale, dry_run = dry_run)

    iterator = (
        tqdm(src_keys, desc = f"↪ {dst_prefix}", unit = "obj")
        if src_keys else src_keys
    )
    for src_key in iterator:
        rel = src_key[len(src_prefix) :]
        dst_key = f"{dst_prefix}{rel}"
        if dry_run:
            print(f"[dry-run] copy s3://{bucket}/{src_key} → s3://{bucket}/{dst_key}")
            continue
        client.copy_object(
            Bucket = bucket,
            Key = dst_key,
            CopySource = {"Bucket": bucket, "Key": src_key},
            ACL = "bucket-owner-full-control",
            MetadataDirective = "COPY",
        )

    return {
        "src_prefix": src_prefix,
        "dst_prefix": dst_prefix,
        "copied": len(src_keys),
        "deleted": len(stale),
    }

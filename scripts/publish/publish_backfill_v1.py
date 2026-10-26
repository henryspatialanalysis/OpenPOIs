"""
Publish one stable-id backfill release as ``<release>-v1`` on Source Cooperative.

Run after ``scripts/conflation/backfill_stable_ids.py`` has written
``<work-dir>/<release>/v1/``. The v1 folder is complete: it holds

- ``conflated-parquet/``: the rewritten partitions (uploaded);
- ``ids/id_registry.parquet``, ``ids/id_changes.parquet``,
  ``ids/id_migration_v0_v1.parquet`` (uploaded);
- ``conflated-pmtiles/``: re-uploaded from ``--conflated-pmtiles`` when given
  (2026-10-22, rebuilt with the new ids), else copied from v0;
- every other v0 object (OSM parquet and PMTiles, Overture PMTiles), copied
  server-side. The proxy cannot copy objects over 5 GB, so the Overture PMTiles
  of 2026-10-01 / 2026-10-22 must be passed as ``--local FILE=PATH`` (a local
  download of the v0 object) and are uploaded;
- ``README.md``: the v0 README under a v1 header.

The v0 README gains one pointer line to v1. ``--mirror-latest`` then mirrors the
v1 folder to ``latest/`` (large files uploaded from the same local copies), and
``--update-top-level`` uploads CHANGELOG.md and the top-level README.

Usage:
    source-coop login; scripts/remote/openpois-remote.sh creds put
    python -u scripts/publish/publish_backfill_v1.py --release 2026-04-23
    python -u scripts/publish/publish_backfill_v1.py --release 2026-10-22 \\
        --conflated-pmtiles ~/data/openpois/conflation/20261022/conflated.pmtiles \\
        --local overture-pmtiles/overture.pmtiles=/path/to/overture.pmtiles \\
        --mirror-latest --update-top-level
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config_versioned import Config

from openpois.io.credentials import load_source_coop_credentials
from openpois.io.source_coop import (
    make_client,
    mirror_prefix,
    upload_bytes,
    upload_directory,
    upload_file,
)
from openpois.publish.build_readme import build_top_readme

CONFIG_PATH = Path("~/repos/openpois/config.yaml").expanduser()
MAX_COPY_BYTES = 5 * 1024 ** 3
ID_FILES = ("id_registry.parquet", "id_changes.parquet", "id_migration_v0_v1.parquet")
POINTER_MARK = "<!-- stable-id-v1 -->"


def v1_header(release: str) -> str:
    return f"""# OpenPOIs {release}-v1

This is the **stable-id** re-issue of [{release}-v0](../{release}-v0/). The only
changes from v0 are:

- `unified_id` is now an opaque id of the form `opoi_` + 16 characters that stays
  the same for a place across releases (see the top-level README). The v0 ids are
  mapped to v1 ids in `ids/id_migration_v0_v1.parquet`.
- `osm_id` is stored as int64 (it was a double in releases up to 2026-09-02).
- `ids/id_registry.parquet` (every source key seen up to this release and its
  id) and `ids/id_changes.parquet` (ids minted, merged, split, retired or
  revived since the previous release) are new.

Everything else, including every confidence value, is identical to v0.

---

"""


def _read_text(client, bucket: str, key: str) -> str:
    return client.get_object(Bucket = bucket, Key = key)["Body"].read().decode("utf-8")


def _objects(client, bucket: str, prefix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket = bucket, Prefix = prefix,
    ):
        for obj in page.get("Contents", []) or []:
            out[obj["Key"][len(prefix):]] = obj["Size"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--release", required = True, help = "YYYY-MM-DD")
    parser.add_argument(
        "--work-dir", type = Path,
        default = Path("~/data/openpois/conflation_backfill").expanduser(),
    )
    parser.add_argument("--conflated-pmtiles", type = Path, default = None)
    parser.add_argument(
        "--local", action = "append", default = [],
        help = "REL=PATH: upload PATH as v1/REL instead of copying v0/REL.",
    )
    parser.add_argument("--skip-parquet", action = "store_true")
    parser.add_argument("--mirror-latest", action = "store_true")
    parser.add_argument("--update-top-level", action = "store_true")
    parser.add_argument("--dry-run", action = "store_true")
    args = parser.parse_args()

    config = Config(str(CONFIG_PATH))
    bucket = config.get("publish", "bucket")
    repo = config.get("publish", "repo_prefix").rstrip("/")
    creds_file = Path(config.get("publish", "credentials_file")).expanduser()
    client = make_client(load_source_coop_credentials(creds_file))

    v0 = f"{repo}/{args.release}-v0/"
    v1 = f"{repo}/{args.release}-v1/"
    local_v1 = args.work_dir / args.release / "v1"
    local_files = {}
    for item in args.local:
        rel, path = item.split("=", 1)
        local_files[rel] = Path(path).expanduser()
    if args.conflated_pmtiles is not None:
        local_files["conflated-pmtiles/conflated.pmtiles"] = args.conflated_pmtiles
    for rel, path in local_files.items():
        if not path.exists():
            raise SystemExit(f"--local {rel}: {path} not found")

    src = _objects(client, bucket, v0)
    if not src:
        raise SystemExit(f"Nothing under s3://{bucket}/{v0}")
    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] {v0} -> {v1}")

    # 1. Rewritten parquet and the id files.
    if not args.skip_parquet:
        upload_directory(
            client = client, local_dir = local_v1 / "conflated-parquet",
            bucket = bucket, key_prefix = f"{v1}conflated-parquet",
            dry_run = args.dry_run,
        )
    for name in ID_FILES:
        upload_file(
            client = client, local_path = local_v1 / name, bucket = bucket,
            key = f"{v1}ids/{name}", dry_run = args.dry_run,
        )

    # 2. Everything else from v0: local re-uploads, else server-side copies.
    for rel, size in sorted(src.items()):
        if rel.startswith("conflated-parquet/") or rel == "README.md":
            continue
        if rel in local_files:
            upload_file(
                client = client, local_path = local_files[rel], bucket = bucket,
                key = f"{v1}{rel}", dry_run = args.dry_run,
                content_type = (
                    "application/json" if rel.endswith(".json")
                    else "application/octet-stream"
                ),
            )
            continue
        if size > MAX_COPY_BYTES:
            raise SystemExit(
                f"{rel} is {size / 1e9:.1f} GB, over the proxy's copy limit; "
                f"pass --local {rel}=<local copy>"
            )
        print(f"  copy {rel} ({size / 1e9:.2f} GB)")
        if not args.dry_run:
            client.copy_object(
                Bucket = bucket, Key = f"{v1}{rel}",
                CopySource = {"Bucket": bucket, "Key": f"{v0}{rel}"},
                ACL = "bucket-owner-full-control", MetadataDirective = "COPY",
            )

    # 3. READMEs.
    v0_readme = _read_text(client, bucket, f"{v0}README.md")
    upload_bytes(
        client = client,
        data = (v1_header(args.release) + v0_readme).encode("utf-8"),
        bucket = bucket, key = f"{v1}README.md",
        content_type = "text/markdown; charset=utf-8", dry_run = args.dry_run,
    )
    if POINTER_MARK not in v0_readme:
        pointer = (
            f"{POINTER_MARK}\n> **Superseded by [{args.release}-v1](../{args.release}-v1/)**, "
            "which is identical except for release-stable `unified_id` values "
            "(v0 -> v1 map in `ids/id_migration_v0_v1.parquet`).\n\n"
        )
        upload_bytes(
            client = client, data = (pointer + v0_readme).encode("utf-8"),
            bucket = bucket, key = f"{v0}README.md",
            content_type = "text/markdown; charset=utf-8", dry_run = args.dry_run,
        )

    # 4. Check v1 against v0 (every v0 path present, plus the id files).
    if not args.dry_run:
        dst = _objects(client, bucket, v1)
        missing = sorted(set(src) - set(dst))
        if missing:
            raise SystemExit(f"v1 is missing {len(missing)} v0 path(s): {missing[:5]}")
        print(f"  v1 holds {len(dst)} objects ({len(src)} in v0)")

    # 5. latest/ and top level.
    if args.mirror_latest:
        summary = mirror_prefix(
            client = client, bucket = bucket, src_prefix = v1,
            dst_prefix = f"{repo}/latest/", dry_run = args.dry_run,
            local_files = local_files,
        )
        print(f"  latest/: copied {summary['copied']}, deleted {summary['deleted']}")
    if args.update_top_level:
        upload_bytes(
            client = client,
            data = (CONFIG_PATH.parent / "CHANGELOG.md").read_bytes(),
            bucket = bucket, key = f"{repo}/CHANGELOG.md",
            content_type = "text/markdown; charset=utf-8", dry_run = args.dry_run,
        )
        upload_bytes(
            client = client, data = build_top_readme().encode("utf-8"),
            bucket = bucket, key = f"{repo}/README.md",
            content_type = "text/markdown; charset=utf-8", dry_run = args.dry_run,
        )
    print("PUBLISH DONE")


if __name__ == "__main__":
    main()

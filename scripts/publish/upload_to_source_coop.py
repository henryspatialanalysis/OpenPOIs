"""
Publish a versioned OpenPOIs release to Source Cooperative.

Reads the local OSM snapshot + conflated outputs from paths resolved via
``config.yaml`` and uploads them under the configured version folder on
Source Cooperative, alongside a freshly generated per-version README.

Remote layout (see docs/data-versioning.md). Writes go through the Source Coop
data proxy, where the BUCKET is the account and the repository is the first key
segment:

    s3://henryspatialanalysis/openpois/          (endpoint https://data.source.coop)
        <YYYY-MM-DD-vN>/
            README.md
            osm-parquet/geohash_prefix=*/part-*.parquet
            osm-pmtiles/osm.pmtiles
            conflated-parquet/geohash_prefix=*/part-*.parquet
            conflated-pmtiles/conflated.pmtiles
        latest/
            (server-side mirror of the most recently published version)

Public *reads* are unaffected and still work anonymously against the legacy flat
bucket ``us-west-2.opendata.source.coop`` with ``henryspatialanalysis/openpois/``
keys, which is what the published README's quickstart examples use.

Credentials
-----------
Source Coop uses OIDC/STS: the ``source-coop`` CLI mints short-lived credentials
after a browser login, and ``openpois.io.credentials`` reads them via
``source-coop creds``. The CLI is auth-only; this script still does the transfer.

    source-coop login          # once per session, ~1 hour of validity
    python -u scripts/publish/upload_to_source_coop.py

A full release is ~7 GB, so check the expiry ``source-coop creds`` reports before
starting. On ``ExpiredToken`` mid-transfer, re-login and resume with the
``--skip-*`` flags; ``latest/`` mirrors last so it is never left half-updated.

``.env.json`` at the repo root remains a fallback for a static credential block,
but Source Coop no longer issues keys that way.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config_versioned import Config

from openpois.io.credentials import load_source_coop_credentials
from openpois.io.source_coop import (
    make_client,
    mirror_prefix,
    public_url,
    upload_bytes,
    upload_directory,
    upload_file,
)
from openpois.publish.build_readme import (
    build_top_readme,
    build_version_readme,
    load_license,
)

CONFIG_PATH = Path("~/repos/openpois/config.yaml").expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--skip-osm-parquet", action = "store_true",
        help = "Skip the OSM-only geohash-partitioned parquet upload.",
    )
    parser.add_argument(
        "--skip-conflated-parquet", action = "store_true",
        help = "Skip the conflated geohash-partitioned parquet upload.",
    )
    parser.add_argument(
        "--skip-pmtiles", action = "store_true",
        help = "Skip both PMTiles uploads.",
    )
    parser.add_argument(
        "--update-top-level", action = "store_true",
        help = (
            "Also (re)upload the repo-root README.md and LICENSE. Off by "
            "default because these rarely change."
        ),
    )
    parser.add_argument(
        "--skip-changelog", action = "store_true",
        help = (
            "Skip uploading CHANGELOG.md to the repo top level. Default "
            "is to upload it on every run so the public copy stays in "
            "sync with the latest per-release deltas."
        ),
    )
    parser.add_argument(
        "--skip-latest-mirror", action = "store_true",
        help = (
            "Skip mirroring the uploaded version to {repo_prefix}/latest/. "
            "Enabled by default so /latest always tracks the newest release."
        ),
    )
    parser.add_argument(
        "--dry-run", action = "store_true",
        help = "Print every remote key that would be written and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config(str(CONFIG_PATH))
    version = config.get("versions", "source_coop")
    bucket = config.get("publish", "bucket")
    repo_prefix = config.get("publish", "repo_prefix").rstrip("/")
    creds_file = Path(
        config.get("publish", "credentials_file")
    ).expanduser()

    creds = load_source_coop_credentials(creds_file)
    client = None if args.dry_run else make_client(creds)

    mode = "DRY RUN — no uploads" if args.dry_run else "LIVE UPLOAD"
    print(f"[{mode}] bucket=s3://{bucket}/  prefix={repo_prefix}/  version={version}")
    print()

    version_prefix = f"{repo_prefix}/{version}"

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    if not args.skip_osm_parquet:
        local = config.get_file_path("snapshot_osm", "partitioned")
        upload_directory(
            client = client,
            local_dir = local,
            bucket = bucket,
            key_prefix = f"{version_prefix}/osm-parquet",
            patterns = ("*.parquet",),
            dry_run = args.dry_run,
        )

    if not args.skip_conflated_parquet:
        local = config.get_file_path("conflation", "partitioned")
        upload_directory(
            client = client,
            local_dir = local,
            bucket = bucket,
            key_prefix = f"{version_prefix}/conflated-parquet",
            patterns = ("*.parquet",),
            dry_run = args.dry_run,
        )

    if not args.skip_pmtiles:
        osm_pm = config.get_file_path("snapshot_osm", "pmtiles")
        if osm_pm.exists():
            upload_file(
                client = client,
                local_path = osm_pm,
                bucket = bucket,
                key = f"{version_prefix}/osm-pmtiles/osm.pmtiles",
                dry_run = args.dry_run,
            )
        else:
            print(f"Skipping OSM PMTiles — {osm_pm} not found.")

        conflated_pm = config.get_file_path("conflation", "pmtiles")
        if conflated_pm.exists():
            upload_file(
                client = client,
                local_path = conflated_pm,
                bucket = bucket,
                key = f"{version_prefix}/conflated-pmtiles/conflated.pmtiles",
                dry_run = args.dry_run,
            )
        else:
            print(f"Skipping conflated PMTiles — {conflated_pm} not found.")

    # -------------------------------------------------------------------------
    # Per-version README (always regenerated)
    # -------------------------------------------------------------------------
    readme_text = build_version_readme(
        config = config,
        version_folder = version,
        config_path = CONFIG_PATH,
    )
    upload_bytes(
        client = client,
        data = readme_text.encode("utf-8"),
        bucket = bucket,
        key = f"{version_prefix}/README.md",
        content_type = "text/markdown; charset=utf-8",
        dry_run = args.dry_run,
    )

    # -------------------------------------------------------------------------
    # Mirror the published version to {repo_prefix}/latest/
    # -------------------------------------------------------------------------
    latest_prefix = f"{repo_prefix}/latest"
    if not args.skip_latest_mirror:
        print()
        print(f"Mirroring {version_prefix}/ → {latest_prefix}/ …")
        if args.dry_run:
            print(
                "  [dry-run] skipping remote listing — copy/delete counts "
                "will only reflect real remote state during a live run."
            )
        summary = mirror_prefix(
            client = client,
            bucket = bucket,
            src_prefix = f"{version_prefix}/",
            dst_prefix = f"{latest_prefix}/",
            dry_run = args.dry_run,
        )
        print(
            f"  copied {summary['copied']} object(s), "
            f"deleted {summary['deleted']} stale object(s)."
        )

    # -------------------------------------------------------------------------
    # Top-level CHANGELOG.md (per-release deltas, updated every run by default)
    # -------------------------------------------------------------------------
    if not args.skip_changelog:
        changelog_path = CONFIG_PATH.parent / "CHANGELOG.md"
        if changelog_path.exists():
            upload_bytes(
                client = client,
                data = changelog_path.read_bytes(),
                bucket = bucket,
                key = f"{repo_prefix}/CHANGELOG.md",
                content_type = "text/markdown; charset=utf-8",
                dry_run = args.dry_run,
            )
        else:
            print(
                f"Skipping CHANGELOG.md upload — {changelog_path} not found."
            )

    # -------------------------------------------------------------------------
    # Top-level README + LICENSE (opt-in)
    # -------------------------------------------------------------------------
    if args.update_top_level:
        upload_bytes(
            client = client,
            data = build_top_readme().encode("utf-8"),
            bucket = bucket,
            key = f"{repo_prefix}/README.md",
            content_type = "text/markdown; charset=utf-8",
            dry_run = args.dry_run,
        )
        upload_bytes(
            client = client,
            data = load_license().encode("utf-8"),
            bucket = bucket,
            key = f"{repo_prefix}/LICENSE",
            content_type = "text/plain; charset=utf-8",
            dry_run = args.dry_run,
        )

    print()
    print(f"Version landing page: https://source.coop/{repo_prefix}/")
    print(f"Version data root:    {public_url(f'{version_prefix}/')}")
    if not args.skip_latest_mirror:
        print(f"Latest data root:     {public_url(f'{latest_prefix}/')}")


if __name__ == "__main__":
    main()

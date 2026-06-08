#!/usr/bin/env python
"""
Apply a fitted ``constant_breakpoint`` turnover model to the OSM POI snapshot.

The breakpoint model's hazard depends on the tag's **age** (years since its
current value was established), so — unlike the memoryless constant model — the
turnover probability over the ``[last_edited, now]`` window must integrate the
hazard at the POI's true age. We compute it in closed form from the cumulative
hazard

    Λ(a) = λ_1·min(a, t_B) + λ_2·max(0, a − t_B)

    a_last_edit = (last_edited − tag_established) / year
    a_now       = (now         − tag_established) / year
    P(turnover) = 1 − exp( −( Λ(a_now) − Λ(a_last_edit) ) )

propagated over the posterior draws of (λ_1, λ_2, t_B) for the mean / interval.
``tag_established`` is joined per element from ``osm_current_tag.parquet`` (built
by ``add_turnover_columns.py``); POIs with no history match fall back to
``a_last_edit = 0`` (treat the last edit as having established the tag), which
recovers the plain ``1 − exp(−Λ(elapsed))`` form.

Output columns appended to the snapshot:
  p_turnover_mean / p_turnover_lower / p_turnover_upper  — P(change since last edit)
  conf_mean / conf_lower / conf_upper                    — 1 − P (interval flips)
  t2_years            — years since last OSM edit
  tag_age_years       — current tag age a_now (NaN-safe; 0-anchored if unmatched)
  matched_history     — True where tag_established came from history
  model_version       — the breakpoint model version used

Usage:
    python scripts/osm_snapshot/apply_model_breakpoint.py \
        --model-version 2026-06-08-breakpoint-test \
        --current-tag-version 20260608
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from config_versioned import Config


config = Config("~/repos/openpois/config.yaml")

SNAPSHOT_PATH = config.get_file_path("snapshot_osm", "snapshot")
MODEL_BASE = Path(config.get_dir_path("model_output")).parent

SECONDS_PER_YEAR = 365.0 * 86_400.0
BATCH_ROWS = 200_000
ROW_GROUP_SIZE = 50_000
MAX_DRAWS = 1_000          # thin posterior draws to bound the (N × draws) work


def load_breakpoint_draws(model_dir: Path) -> dict[str, np.ndarray]:
    """Load thinned posterior draws of λ_1, λ_2, t_B from param_draws.parquet."""
    path = model_dir / "param_draws.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — fit with save_full_model=true so the breakpoint "
            "draws are persisted."
        )
    df = pd.read_parquet(path, columns = ["lambda_1", "lambda_2", "t_breakpoint"])
    if len(df) > MAX_DRAWS:
        step = len(df) // MAX_DRAWS
        df = df.iloc[::step].iloc[:MAX_DRAWS]
    return {
        "lambda_1": df["lambda_1"].to_numpy(np.float64),
        "lambda_2": df["lambda_2"].to_numpy(np.float64),
        "t_breakpoint": df["t_breakpoint"].to_numpy(np.float64),
    }


def cumulative_hazard(age: np.ndarray, draws: dict[str, np.ndarray]) -> np.ndarray:
    """Λ(age) over draws → (n_rows, n_draws). age is (n_rows,)."""
    a = age[:, None]
    lam1 = draws["lambda_1"][None, :]
    lam2 = draws["lambda_2"][None, :]
    t_b = draws["t_breakpoint"][None, :]
    return lam1 * np.minimum(a, t_b) + lam2 * np.maximum(0.0, a - t_b)


def turnover_stats(
    a_last_edit: np.ndarray, a_now: np.ndarray, draws: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean / 2.5% / 97.5% of P(turnover) over the [last_edit, now] window."""
    integrated = (
        cumulative_hazard(a_now, draws) - cumulative_hazard(a_last_edit, draws)
    )
    p = 1.0 - np.exp(-integrated)
    return (
        p.mean(axis = 1),
        np.quantile(p, 0.025, axis = 1),
        np.quantile(p, 0.975, axis = 1),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Rate the OSM snapshot with a fitted constant_breakpoint model."
    )
    parser.add_argument("--model-version", required = True)
    parser.add_argument(
        "--current-tag-version", required = True,
        help = "osm_data version holding osm_current_tag.parquet (tag_established).",
    )
    parser.add_argument(
        "--output", default = None,
        help = "Output parquet path (default: <model_dir>/osm_snapshot_turnover.parquet).",
    )
    parser.add_argument(
        "--test", action = "store_true",
        help = "Process only the first 10,000 snapshot rows.",
    )
    args = parser.parse_args()

    model_dir = MODEL_BASE / args.model_version
    draws = load_breakpoint_draws(model_dir)
    print(
        f"Loaded {len(draws['lambda_1'])} draws | "
        f"lambda_1 mean {draws['lambda_1'].mean():.4f}, "
        f"lambda_2 mean {draws['lambda_2'].mean():.4f}, "
        f"t_B mean {draws['t_breakpoint'].mean():.4f} yr"
    )

    cur_path = config.get_dir_path(
        "osm_data", custom_version = args.current_tag_version
    ) / "osm_current_tag.parquet"
    print(f"Loading tag_established lookup from {cur_path} ...")
    cur = pd.read_parquet(cur_path)
    cur["tag_established"] = pd.to_datetime(cur["tag_established"], utc = True)
    # Series indexed by (osm_type, id) → tag_established (ns since epoch, UTC).
    establ = pd.Series(
        cur["tag_established"].to_numpy("datetime64[ns]"),
        index = pd.MultiIndex.from_arrays(
            [cur["osm_type"].to_numpy(), cur["id"].to_numpy()],
            names = ["osm_type", "id"],
        ),
    )
    print(f"  {len(establ):,} elements with a history establishment time.")

    output_path = (
        Path(args.output) if args.output
        else model_dir / "osm_snapshot_turnover.parquet"
    )

    pf = pq.ParquetFile(SNAPSHOT_PATH)
    n_total = pf.metadata.num_rows
    print(f"Rating {n_total:,} POIs from {SNAPSHOT_PATH} → {output_path}")
    now = pd.Timestamp.now(tz = "UTC")

    out_fields = [
        pa.field("osm_id", pa.int64()),
        pa.field("osm_type", pa.string()),
        pa.field("name", pa.string()),
        pa.field("last_edited", pa.timestamp("us", tz = "UTC")),
        pa.field("t2_years", pa.float64()),
        pa.field("tag_age_years", pa.float64()),
        pa.field("matched_history", pa.bool_()),
        pa.field("p_turnover_mean", pa.float64()),
        pa.field("p_turnover_lower", pa.float64()),
        pa.field("p_turnover_upper", pa.float64()),
        pa.field("conf_mean", pa.float64()),
        pa.field("conf_lower", pa.float64()),
        pa.field("conf_upper", pa.float64()),
        pa.field("model_version", pa.string()),
    ]
    out_schema = pa.schema(out_fields)

    have_name = "name" in set(pf.schema_arrow.names)
    read_cols = ["osm_id", "osm_type", "last_edited"] + (["name"] if have_name else [])

    output_path.parent.mkdir(parents = True, exist_ok = True)
    n_written = 0
    n_matched = 0
    with pq.ParquetWriter(output_path, out_schema, compression = "zstd") as writer:
        batches = (
            [next(pf.iter_batches(batch_size = 10_000, columns = read_cols))]
            if args.test
            else pf.iter_batches(batch_size = BATCH_ROWS, columns = read_cols)
        )
        for batch in batches:
            df = pa.Table.from_batches([batch]).to_pandas()
            last_edited = pd.to_datetime(df["last_edited"], utc = True)

            # Join tag_established by (osm_type, osm_id); unmatched → NaT.
            keys = pd.MultiIndex.from_arrays(
                [df["osm_type"].to_numpy(), df["osm_id"].to_numpy()],
                names = ["osm_type", "id"],
            )
            tag_est = pd.Series(
                establ.reindex(keys).to_numpy(), index = df.index
            )
            tag_est = pd.to_datetime(tag_est, utc = True)
            matched = tag_est.notna().to_numpy()
            # Fallback: unmatched POIs anchor the tag clock at their last edit.
            tag_est = tag_est.fillna(last_edited)

            a_last_edit = np.clip(
                (last_edited - tag_est).dt.total_seconds().to_numpy()
                / SECONDS_PER_YEAR,
                0.0, None,
            )
            elapsed = (now - last_edited).dt.total_seconds().to_numpy()
            t2_years = elapsed / SECONDS_PER_YEAR
            a_now = np.maximum(
                (now - tag_est).dt.total_seconds().to_numpy() / SECONDS_PER_YEAR,
                a_last_edit,
            )

            p_mean, p_lo, p_hi = turnover_stats(a_last_edit, a_now, draws)

            n = len(df)
            name_ser = (
                df["name"] if have_name
                else pd.Series([None] * n, dtype = "object")
            )
            out = pa.table({
                "osm_id": pa.array(df["osm_id"].to_numpy(), pa.int64()),
                "osm_type": pa.array(df["osm_type"].astype(str).to_numpy()),
                # from_pandas → pandas NA / NaN become Arrow nulls.
                "name": pa.array(name_ser, type = pa.string(), from_pandas = True),
                "last_edited": pa.array(
                    last_edited, type = pa.timestamp("us", tz = "UTC"),
                    from_pandas = True,
                ),
                "t2_years": t2_years,
                "tag_age_years": a_now,
                "matched_history": matched,
                "p_turnover_mean": p_mean,
                "p_turnover_lower": p_lo,
                "p_turnover_upper": p_hi,
                "conf_mean": 1.0 - p_mean,
                "conf_lower": 1.0 - p_hi,
                "conf_upper": 1.0 - p_lo,
                "model_version": np.full(n, args.model_version, dtype = object),
            }, schema = out_schema)
            writer.write_table(out, row_group_size = ROW_GROUP_SIZE)
            n_written += n
            n_matched += int(matched.sum())
            print(f"  {n_written:,}/{n_total:,} rated", flush = True)

    print(
        f"\nDone. Rated {n_written:,} POIs → {output_path}\n"
        f"History-matched tag age: {n_matched:,} "
        f"({100 * n_matched / max(n_written, 1):.1f}%); "
        f"{n_written - n_matched:,} used the last-edit fallback."
    )

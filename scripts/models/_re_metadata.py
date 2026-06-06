"""
Shared builder for ``RandomEffectsModel`` metadata from ``config.yaml``.

Imported by both ``osm_turnover.py`` (in-sample fit) and ``osm_turnover_cv.py``
(out-of-sample cross-validation) so the two entry points assemble identical
metadata from the same ``osm_turnover_model.random_effects`` block — every
prior/toggle flows from config, no magic numbers in the scripts.
"""
from __future__ import annotations

from collections.abc import Iterable

_KEEP = "__keep__"


def build_random_effects_metadata(
    config,
    enabled_terms: Iterable[str] | None = None,
    delta_group: str | None = _KEEP,
) -> dict:
    """Assemble the ``random_effects`` model metadata from config.

    Args:
        config: a ``config_versioned.Config`` instance.
        enabled_terms: term names to include. ``None`` (default) honours each
            term's ``enabled`` flag in config; otherwise only the named terms
            are included regardless of their ``enabled`` flag.
        delta_group: ``"__keep__"`` (default) uses the configured
            ``random_effects.delta_group``; ``None`` forces a single global δ;
            any other string overrides the δ grouping column.

    Returns:
        Metadata dict suitable for ``RandomEffectsModel(metadata=...)``.
    """
    re_cfg = config.get("osm_turnover_model", "random_effects")
    enabled_set = None if enabled_terms is None else set(enabled_terms)
    terms: dict[str, dict] = {}
    for name, tcfg in re_cfg["terms"].items():
        if enabled_set is None:
            if not tcfg.get("enabled"):
                continue
        elif name not in enabled_set:
            continue
        if name == "amenity_msa":
            terms[name] = {
                "columns": list(tcfg["columns"]),
                "var_prior": tuple(tcfg["var_prior"]),
                "min_count": int(re_cfg.get("interaction_min_count", 100)),
            }
        elif name == "urbanicity":
            terms[name] = {
                "column": tcfg["column"],
                "prior": tuple(tcfg["prior"]),
            }
        else:
            terms[name] = {
                "column": tcfg["column"],
                "var_prior": tuple(tcfg["var_prior"]),
            }
    dg = re_cfg.get("delta_group") if delta_group == _KEEP else delta_group
    metadata = {
        "dt_col": "tag_years",
        "terms": terms,
        "delta_group": dg,
    }
    ldp = config.get(
        "osm_turnover_model", "logit_delta_prior", fail_if_none = False
    )
    if ldp is not None:
        metadata["logit_delta_prior"] = tuple(ldp)
    ldvp = config.get(
        "osm_turnover_model", "logit_delta_var_prior", fail_if_none = False
    )
    if ldvp is not None:
        metadata["logit_delta_var_prior"] = tuple(ldvp)
    return metadata

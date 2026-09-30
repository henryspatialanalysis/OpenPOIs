"""Tests for the Bayesian monotone-spline calibration prototype.

Each test pins one property the design doc
(``.claude/plans/bayesian-monotone-calibration.md``, §4) relies on:

- the curves are monotone and inside (L, U) for any parameter value (M7-M10)
- the 2-D max recursion is a bijection onto doubly-increasing matrices, so it
  imposes no sign on the interaction (M9)
- constant log-slopes on equal spacing give an additive linear surface
- the log-probabilities stay finite at the range bounds (M4)
- the three likelihood arms reduce to the plain Bernoulli where they should, and
  the jitted log density matches a numpy reference in value and gradient
- arm C's fixed silver-label rates (its default data layer, decision 21): the
  Hajek design-weighted, Jeffreys-smoothed segment x verdict gold rates, their
  holdout and multi-round pooling, and the fractional-label identity (M15c')
- the knot rule (decision 5)
- CV folds are identical to the production cross-fit's
- a short fit recovers a known curve
- the ``jax_core`` sampler's default path is unchanged by the new passthrough
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, log_expit

from openpois.conflation import calibration_bayes as cb
from openpois.conflation import calibration_fit as cf
from openpois.models.jax_core import nuts_sample_multichain


def _synthetic_rows(n_per_segment = 600, seed = 0) -> pd.DataFrame:
    """A two-phase validation table with known monotone truth curves."""
    rng = np.random.default_rng(seed)
    frames = []
    for segment in cb.SEGMENT_ORDER:
        n = n_per_segment
        osm = np.clip(rng.beta(5.0, 1.5, n), 0.0, 1.0)
        overture = rng.choice([0.919912, 0.990219, 0.5], n, p = [0.3, 0.2, 0.5])
        cont = rng.uniform(0.05, 1.0, n)
        overture = np.where(overture == 0.5, cont, overture)
        if segment == "overture":
            truth = -1.0 + 3.0 * overture
            osm = np.full(n, np.nan)
        elif segment == "osm":
            truth = -1.5 + 3.5 * osm
            overture = np.full(n, np.nan)
        else:
            truth = -2.0 + 2.5 * osm + 2.5 * overture
        y = rng.uniform(size = n) < expit(truth)
        u = rng.uniform(size = n)
        verdict = np.where(
            y, np.where(u < 0.75, "exists", np.where(u < 0.78, "gone",
                                                      "unverifiable")),
            np.where(u < 0.08, "exists", np.where(u < 0.70, "gone",
                                                   "unverifiable")),
        )
        confidence = rng.choice(["high", "medium", "low"], n, p = [0.6, 0.3, 0.1])
        rate = {"exists": 0.12, "gone": 0.37, "unverifiable": 1.0}
        gold = np.zeros(n, dtype = bool)
        for v, r in rate.items():
            idx = np.flatnonzero(verdict == v)
            take = rng.choice(idx, int(round(r * len(idx))), replace = False)
            gold[take] = True
        raw = np.where(np.isnan(osm), overture,
                       np.where(np.isnan(overture), osm,
                                0.588 * osm + 0.412 * overture))
        frames.append(pd.DataFrame({
            "segment": segment, "stratum": segment,
            "osm_score": osm, "overture_score": overture, "raw_score": raw,
            "llm_verdict": verdict, "llm_confidence": confidence,
            "gold": gold, "y": np.where(gold, y.astype(float), np.nan),
            "truth_logit": truth,
        }))
    return cb.usable_rows(pd.concat(frames, ignore_index = True))


@pytest.fixture(scope = "module")
def rows():
    return _synthetic_rows()


def _random_params(prepared, rng, scale = 1.5):
    template = cb.parameter_template(prepared)
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(rng.normal(0.0, scale, np.shape(x))), template
    )


def test_curves_monotone_and_bounded(rows):
    prepared = cb.prepare_data(rows, cb.ModelSpec())
    rng = np.random.default_rng(1)
    draws = [_random_params(prepared, rng) for _ in range(40)]
    stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *draws)
    grid = np.linspace(0.0, 1.0, 10_001)
    for segment in cb.ONE_D_SEGMENTS:
        m = cb.curve_draws(stacked, prepared, segment, osm = grid, overture = grid)
        # Rounding noise only: fp32 unless the caller enabled x64.
        tol = 50 * np.finfo(m.dtype).eps
        assert np.all(np.diff(m, axis = 1) >= -tol)
        assert np.all((m >= expit(cb.L_BOUND) - tol)
                      & (m <= expit(cb.U_BOUND) + tol))
    axis = np.linspace(0.0, 1.0, 401)
    xx, yy = np.meshgrid(axis, axis, indexing = "ij")
    m = cb.curve_draws(stacked, prepared, "matched", osm = xx.ravel(),
                       overture = yy.ravel()).reshape(-1, 401, 401)
    tol = 50 * np.finfo(m.dtype).eps
    assert np.all(np.diff(m, axis = 1) >= -tol)
    assert np.all(np.diff(m, axis = 2) >= -tol)
    assert np.all((m >= expit(cb.L_BOUND) - tol)
                  & (m <= expit(cb.U_BOUND) + tol))


def test_max_recursion_is_a_bijection():
    rng = np.random.default_rng(2)
    n_x, n_y = 6, 5
    hx, hy = rng.uniform(0.05, 0.3, n_x - 1), rng.uniform(0.05, 0.3, n_y - 1)
    # A random strictly doubly-increasing matrix, including a substitutive
    # (negative cross-difference) pattern that a Kronecker cumulative sum
    # cannot represent.
    base = np.add.outer(np.cumsum(rng.uniform(0.1, 1.0, n_x)),
                        np.cumsum(rng.uniform(0.1, 1.0, n_y)))
    target = base - 0.05 * np.multiply.outer(np.arange(n_x), np.arange(n_y))
    assert np.all(np.diff(target, axis = 0) > 0)
    assert np.all(np.diff(target, axis = 1) > 0)
    cross = np.diff(np.diff(target, axis = 0), axis = 1)
    assert np.any(cross < 0)
    alpha, gamma = cb.invert_coefficients_2d(target, hx, hy)
    rebuilt = cb.coefficients_2d_tilde(jnp.asarray(alpha), jnp.asarray(gamma),
                                       jnp.asarray(hx), jnp.asarray(hy))
    np.testing.assert_allclose(np.asarray(rebuilt), target, rtol = 1e-6,
                               atol = 1e-6)


def test_constant_log_slope_gives_additive_linear_surface():
    n_x, n_y, h, b, alpha = 5, 4, 0.2, 1.7, -0.3
    gamma = jnp.full((n_x, n_y), np.log(b))
    c = np.asarray(cb.coefficients_2d_tilde(
        jnp.asarray(alpha), gamma, jnp.full(n_x - 1, h), jnp.full(n_y - 1, h)
    ))
    expected = alpha + b * h * np.add.outer(np.arange(n_x), np.arange(n_y))
    np.testing.assert_allclose(c, expected, rtol = 1e-6)


def test_log_probabilities_finite_at_bounds():
    f = jnp.asarray([cb.L_BOUND, 0.0, cb.U_BOUND])
    lp1, lp0 = jax.nn.log_sigmoid(f), jax.nn.log_sigmoid(-f)
    assert np.all(np.isfinite(np.asarray(lp1)))
    assert np.all(np.isfinite(np.asarray(lp0)))
    np.testing.assert_allclose(np.asarray(lp1), log_expit(np.asarray(f)),
                               rtol = 1e-6)
    np.testing.assert_allclose(np.asarray(lp1)[-1], np.log(1 - 1e-3), rtol = 1e-5)


def _bernoulli(params, prepared, labels):
    data = prepared.to_jax()
    coefficients = cb.curve_coefficients(params, prepared.geometry, prepared.spec)
    logits = cb.segment_logits(coefficients, data)
    out = {}
    for segment, f in logits.items():
        f = np.asarray(f)
        y = labels[segment]
        out[segment] = y * log_expit(f) + (1 - y) * log_expit(-f)
    return out


def test_arm_a_reduces_to_bernoulli_under_identity_measurement():
    rows = _synthetic_rows(seed = 3)
    # Deterministic verdicts: exists:high iff y = 1, gone:high iff y = 0.
    truth = np.asarray(rows["truth_logit"] > 0)
    rows = rows.assign(llm_verdict = np.where(truth, "exists", "gone"),
                       llm_confidence = "high",
                       y = np.where(rows["gold"], truth.astype(float), np.nan))
    spec = cb.ModelSpec(arm = "A", differential = False)
    prepared = cb.prepare_data(rows, spec)
    params = _random_params(prepared, np.random.default_rng(4), scale = 0.5)
    gone_high = spec.class_levels.index("gone:high")
    # Full class logits: y = 1 puts all mass on exists:high, y = 0 on gone:high.
    full = np.full((2, len(spec.class_levels)), -60.0)
    full[1, 0] = 60.0
    full[0, gone_high] = 60.0
    free = (full - full.mean(axis = 1, keepdims = True)) @ cb.helmert_basis(
        len(spec.class_levels))
    params["psi"] = jnp.broadcast_to(jnp.asarray(free), params["psi"].shape)
    pointwise = cb.pointwise_log_likelihood(params, prepared.to_jax(),
                                            prepared.geometry, spec)
    labels = {s: (rows.loc[rows["segment"] == s, "truth_logit"] > 0)
              .to_numpy(dtype = float) for s in cb.SEGMENT_ORDER}
    expected = _bernoulli(params, prepared, labels)
    for segment in cb.SEGMENT_ORDER:
        np.testing.assert_allclose(np.asarray(pointwise[segment]),
                                   expected[segment], atol = 1e-6)


def test_arm_c_reduces_to_bernoulli_with_perfect_silver_labels():
    rows = _synthetic_rows(seed = 5)
    truth = np.asarray(rows["truth_logit"] > 0)
    rows = rows.assign(llm_verdict = np.where(truth, "exists", "gone"),
                       y = np.where(rows["gold"], truth.astype(float), np.nan))
    spec = cb.ModelSpec(arm = "C", label_noise = "symmetric")
    prepared = cb.prepare_data(rows, spec)
    params = _random_params(prepared, np.random.default_rng(6), scale = 0.5)
    params["logit_beta_label"] = jnp.asarray(60.0)
    pointwise = cb.pointwise_log_likelihood(params, prepared.to_jax(),
                                            prepared.geometry, spec)
    labels = {s: (rows.loc[rows["segment"] == s, "truth_logit"] > 0)
              .to_numpy(dtype = float) for s in cb.SEGMENT_ORDER}
    expected = _bernoulli(params, prepared, labels)
    for segment in cb.SEGMENT_ORDER:
        np.testing.assert_allclose(np.asarray(pointwise[segment]),
                                   expected[segment], atol = 1e-6)


def _numpy_arm_a(params, prepared):
    """Independent numpy implementation of (M12)-(M15)."""
    spec = prepared.spec
    data = prepared.segments
    coefficients = cb.curve_coefficients(params, prepared.geometry, spec)
    logits = cb.segment_logits(coefficients, prepared.to_jax())
    total = 0.0
    p = {k: np.asarray(v, dtype = float) for k, v in params.items()}
    for g, segment in enumerate(cb.SEGMENT_ORDER):
        seg, f = data[segment], np.asarray(logits[segment], dtype = float)
        basis = cb.helmert_basis(p["psi"].shape[-1] + 1)
        psi, beta = p["psi"][g] @ basis.T, p["beta"][g] @ basis.T
        for i in range(len(f)):
            eta = psi + beta * seg["r"][i]
            log_theta = eta - np.logaddexp.reduce(eta, axis = 1, keepdims = True)
            lt = log_theta[:, seg["cls"][i]]
            lp1, lp0 = log_expit(f[i]), log_expit(-f[i])
            if seg["gold"][i] > 0.5:
                yv = seg["y"][i]
                total += yv * (lp1 + lt[1]) + (1 - yv) * (lp0 + lt[0])
            else:
                total += np.logaddexp(lp1 + lt[1], lp0 + lt[0])
    return total


def test_arm_a_matches_numpy_reference_value_and_gradient():
    rows = _synthetic_rows(n_per_segment = 120, seed = 7)
    prepared = cb.prepare_data(rows, cb.ModelSpec(arm = "A"))
    params = _random_params(prepared, np.random.default_rng(8), scale = 0.4)
    data = prepared.to_jax()

    def loglik(p):
        pointwise = cb.pointwise_log_likelihood(p, data, prepared.geometry,
                                                prepared.spec)
        return sum(jnp.sum(v) for v in pointwise.values())

    np.testing.assert_allclose(float(loglik(params)),
                               _numpy_arm_a(params, prepared), rtol = 1e-5)
    # The jitted log density is the likelihood plus the prior.
    log_density = cb.make_log_density(prepared)
    np.testing.assert_allclose(
        float(log_density(params)),
        float(loglik(params) + cb.log_prior(params, prepared.spec)), rtol = 1e-6,
    )
    grad = jax.grad(loglik)(params)
    for name, index in (("mu", 2), ("psi", (1, 1, 3)), ("beta", (2, 0, 1)),
                        ("z_matched", 5)):
        # eps = 1e-2: in fp32 a smaller step is dominated by rounding noise.
        eps = 1e-2
        up = jax.tree_util.tree_map(lambda x: x, params)
        down = jax.tree_util.tree_map(lambda x: x, params)
        up[name] = up[name].at[index].add(eps)
        down[name] = down[name].at[index].add(-eps)
        numeric = (float(loglik(up)) - float(loglik(down))) / (2 * eps)
        np.testing.assert_allclose(float(grad[name][index]), numeric,
                                   rtol = 1e-2, atol = 1e-2)


def test_knot_rule_handles_atoms_and_gaps():
    rng = np.random.default_rng(9)
    scores = np.concatenate([
        np.full(310, 0.919912), np.full(220, 0.990219),
        rng.uniform(0.9, 1.0, 370), rng.uniform(0.0, 0.9, 100),
    ])
    breaks = cb.knot_breaks(scores, max_gap = 0.2)
    assert breaks[0] == 0.0 and breaks[-1] == 1.0
    assert np.all(np.diff(breaks) > 0)
    assert np.max(np.diff(breaks)) <= 0.2 + 1e-12
    assert np.any(np.isclose(breaks, 0.919912))
    assert np.any(np.isclose(breaks, 0.990219))
    assert len(np.unique(np.round(breaks, 6))) == len(breaks)


def test_fold_assignment_matches_production_cross_fit(rows):
    """Full table, one production cross-fit call per segment (fresh RNG each)."""
    fit_config = cf.FitConfig()
    classes = cb.production_classes(rows, fit_config)
    ours = cb.assign_segment_folds(rows, classes, 10, fit_config.rng_seed)
    for segment in cb.SEGMENT_ORDER:
        mask = (rows["segment"] == segment).to_numpy()
        seg = rows[mask].reset_index(drop = True)
        seg_classes = classes[mask].reset_index(drop = True)
        score = seg["osm_score"] if segment != "overture" else seg["overture_score"]
        index_mode = "average" if segment == "matched" else "pool"
        out = cf.cross_fit_predictions(
            seg.assign(score = score), seg_classes, fit_config, n_folds = 10,
            segment = segment, index_mode = index_mode,
        )
        gold = seg["gold"].to_numpy(dtype = bool)
        assert out["n_gold"] == gold.sum()
        np.testing.assert_array_equal(out["fold"], ours[mask][gold])


def test_segment_logits_match_direct_tensor_product(rows):
    """F3 = sum_jk C_jk B_j(osm) B_k(overture), checked with scipy on an
    asymmetric coefficient matrix (guards the x/y orientation)."""
    from scipy.interpolate import BSpline

    prepared = cb.prepare_data(rows, cb.ModelSpec())
    params = _random_params(prepared, np.random.default_rng(21), scale = 0.8)
    coefficients = cb.curve_coefficients(params, prepared.geometry, prepared.spec)
    logits = cb.segment_logits(coefficients, prepared.to_jax())
    matched = rows[rows["segment"] == "matched"]
    c = np.asarray(coefficients["matched"], dtype = float)
    assert c.shape[0] != c.shape[1]
    tx = cb.clamped_knot_vector(prepared.knots["matched_x"], 2)
    ty = cb.clamped_knot_vector(prepared.knots["matched_y"], 2)
    expected = []
    for x, y in zip(matched["osm_score"], matched["overture_score"]):
        bx = np.array([BSpline.basis_element(tx[j:j + 4], extrapolate = False)(x)
                       for j in range(c.shape[0])])
        by = np.array([BSpline.basis_element(ty[k:k + 4], extrapolate = False)(y)
                       for k in range(c.shape[1])])
        bx, by = np.nan_to_num(bx), np.nan_to_num(by)
        expected.append(bx @ c @ by)
    got = np.asarray(logits["matched"], dtype = float)
    # basis_element is right-open, so compare away from the domain's right end.
    interior = (matched["osm_score"] < 1).to_numpy() & (
        matched["overture_score"] < 1).to_numpy()
    np.testing.assert_allclose(got[interior], np.asarray(expected)[interior],
                               rtol = 1e-4, atol = 1e-4)


@pytest.mark.parametrize("arm", ["A", "C"])
def test_short_fit_recovers_known_curve(arm):
    rows = _synthetic_rows(n_per_segment = 500, seed = 11)
    prepared = cb.prepare_data(rows, cb.ModelSpec(arm = arm))
    result = cb.fit(prepared, num_warmup = 300, num_samples = 300, num_chains = 2,
                    seed = 12)
    osm = rows.loc[rows["segment"] == "osm", "osm_score"]
    median = float(np.median(osm))
    draws = cb.curve_draws(result.draws, prepared, "osm",
                           osm = np.array([median]))
    band = cb.summarize_draws(draws)
    truth = expit(-1.5 + 3.5 * median)
    assert band["lower"][0] - 0.03 <= truth <= band["upper"][0] + 0.03


def test_jax_core_default_path_unchanged():
    """nuts_sample without adaptation_kwargs equals a direct BlackJAX run."""
    import blackjax

    from openpois.models.jax_core import nuts_sample, random_markov_chain

    def log_density(p):
        return -0.5 * jnp.sum(p["x"] ** 2 * jnp.array([1.0, 4.0]))

    start = {"x": jnp.zeros(2)}
    key = jax.random.PRNGKey(3)
    draws, _, warm = nuts_sample(log_density, start, num_warmup = 60,
                                 num_samples = 25, key = key)
    warmup_key, sample_key = jax.random.split(key, 2)
    adaptation = blackjax.window_adaptation(algorithm = blackjax.nuts,
                                            logdensity_fn = log_density)
    (state, params), _ = adaptation.run(rng_key = warmup_key, position = start,
                                        num_steps = 60)
    kernel = blackjax.nuts(logdensity_fn = log_density, **params).step
    states, _ = random_markov_chain(key = sample_key, kernel = kernel,
                                    init_state = state, num_draws = 25)
    np.testing.assert_allclose(np.asarray(draws["x"]),
                               np.asarray(states.position["x"]))
    np.testing.assert_allclose(float(warm["step_size"]), float(params["step_size"]))
    _, _, warm_target = nuts_sample_multichain(
        log_density, start, num_warmup = 60, num_samples = 25, num_chains = 2,
        key = key, adaptation_kwargs = {"target_acceptance_rate": 0.99},
    )
    _, _, warm_default = nuts_sample_multichain(
        log_density, start, num_warmup = 60, num_samples = 25, num_chains = 2,
        key = key,
    )
    assert not np.allclose(np.asarray(warm_default["step_size"]),
                           np.asarray(warm_target["step_size"]))


def test_asymmetric_label_noise_reduces_to_symmetric():
    """Arm C with Se = Sp = beta equals the one-parameter symmetric arm C."""
    rows = _synthetic_rows(seed = 13)
    sym = cb.prepare_data(rows, cb.ModelSpec(arm = "C", label_noise = "symmetric"))
    asym = cb.prepare_data(rows, cb.ModelSpec(arm = "C", label_noise = "asymmetric"))
    params = _random_params(sym, np.random.default_rng(14), scale = 0.5)
    params["logit_beta_label"] = jnp.asarray(2.3)
    params_asym = {k: v for k, v in params.items() if k != "logit_beta_label"}
    params_asym["logit_se"] = jnp.asarray(2.3)
    params_asym["logit_sp"] = jnp.asarray(2.3)
    a = cb.pointwise_log_likelihood(params, sym.to_jax(), sym.geometry, sym.spec)
    b = cb.pointwise_log_likelihood(params_asym, asym.to_jax(), asym.geometry,
                                    asym.spec)
    for segment in cb.SEGMENT_ORDER:
        np.testing.assert_allclose(np.asarray(a[segment]), np.asarray(b[segment]),
                                   atol = 1e-6)
    # "none" treats silver labels as exact: the plain Bernoulli on the label.
    exact = cb.prepare_data(rows, cb.ModelSpec(arm = "C", label_noise = "none"))
    params_none = {k: v for k, v in params.items() if k != "logit_beta_label"}
    c = cb.pointwise_log_likelihood(params_none, exact.to_jax(), exact.geometry,
                                    exact.spec)
    params["logit_beta_label"] = jnp.asarray(60.0)
    d = cb.pointwise_log_likelihood(params, sym.to_jax(), sym.geometry, sym.spec)
    for segment in cb.SEGMENT_ORDER:
        np.testing.assert_allclose(np.asarray(c[segment]), np.asarray(d[segment]),
                                   atol = 1e-5)


def test_fixed_silver_rates_match_hand_computation_and_respect_holdout(rows):
    """silver_label_rates: design-weighted, Jeffreys-smoothed, training gold only."""
    fit_config = cf.FitConfig()
    rates = cb.silver_label_rates(rows, fit_config)
    classes = cb.production_classes(rows, fit_config)
    gold = rows["gold"].to_numpy(dtype = bool)
    seg = "osm"
    mask = (rows["segment"] == seg).to_numpy()
    inclusion = cf.inclusion_by_class(pd.Series(classes[mask]).reset_index(drop = True),
                                      gold[mask])
    w = np.array([inclusion[c]["weight"] if c in inclusion else 0.0
                  for c in classes[mask]]) * gold[mask]
    sel = (rows.loc[mask, "llm_verdict"] == "exists").to_numpy() & (w > 0)
    y = rows.loc[mask, "y"].to_numpy(dtype = float)[sel]
    r = np.sum(w[sel] * y) / np.sum(w[sel])
    ess = np.sum(w[sel]) ** 2 / np.sum(w[sel] ** 2)
    np.testing.assert_allclose(rates[seg]["exists"], (r * ess + 0.5) / (ess + 1))
    # Holding out gold removes it from the rates.
    held = np.zeros(len(rows), dtype = bool)
    held[np.flatnonzero(gold)[::2]] = True
    fewer = cb.silver_label_rates(rows, fit_config, gold_masks = [gold & ~held])
    assert fewer[seg]["n_exists"] < rates[seg]["n_exists"]
    # Pooling a second round adds its gold.
    pooled = cb.silver_label_rates([rows, rows], fit_config)
    assert pooled[seg]["n_exists"] == 2 * rates[seg]["n_exists"]


def test_fixed_rate_arm_c_is_a_fractional_label_bernoulli(rows):
    """With q = 1 / 0 the fixed-rate likelihood is the Bernoulli on the label."""
    spec = cb.ModelSpec(arm = "C")
    assert spec.label_noise == "fixed"
    exact_rates = {s: {"exists": 1.0, "gone": 0.0} for s in cb.SEGMENT_ORDER}
    prepared = cb.prepare_data(rows, spec, silver_rates = exact_rates)
    params = _random_params(prepared, np.random.default_rng(31), scale = 0.5)
    assert "logit_beta_label" not in params
    got = cb.pointwise_log_likelihood(params, prepared.to_jax(), prepared.geometry,
                                      spec)
    labels = {}
    for s in cb.SEGMENT_ORDER:
        r = rows[rows["segment"] == s]
        labels[s] = np.where(r["gold"], r["y"].fillna(0),
                             (r["llm_verdict"] == "exists").astype(float))
    expected = _bernoulli(params, prepared, labels)
    for s in cb.SEGMENT_ORDER:
        silver_unlabelled = (~rows.loc[rows["segment"] == s, "gold"]
                             & (rows.loc[rows["segment"] == s, "llm_verdict"]
                                == "unverifiable")).to_numpy()
        np.testing.assert_allclose(np.asarray(got[s])[~silver_unlabelled],
                                   expected[s][~silver_unlabelled], atol = 1e-5)


def test_pooled_rounds_keep_each_round_design(rows):
    """Pooled rounds (execution log, decision 23) keep each round's design.

    Classes, weights and silver-label rates are computed per round.
    """
    fit_config = cf.FitConfig()
    july = rows.assign(validation_round = "A")
    october = _synthetic_rows(n_per_segment = 400, seed = 5).assign(
        validation_round = "B")
    pooled = cb.usable_rows(pd.concat([july, october], ignore_index = True))
    classes = cb.production_classes(pooled, fit_config)
    # Each round's classes are its own single-round classes, prefixed.
    for round_id, frame in (("A", july), ("B", october)):
        mine = classes[(pooled["validation_round"] == round_id).to_numpy()]
        alone = cb.production_classes(cb.usable_rows(frame), fit_config)
        assert (mine.to_numpy() == (f"{round_id}|" + alone).to_numpy()).all()
    # A single-round table keeps the unprefixed production classes.
    assert not cb.production_classes(july, fit_config).str.contains("|",
                                                                    regex = False).any()
    # Rates from the pooled table equal rates from the per-round frames.
    one_table = cb.silver_label_rates(pooled, fit_config)
    two_frames = cb.silver_label_rates([cb.usable_rows(july),
                                        cb.usable_rows(october)], fit_config)
    for s in cb.SEGMENT_ORDER:
        for key in ("exists", "gone", "n_exists", "n_gone"):
            np.testing.assert_allclose(one_table[s][key], two_frames[s][key])
    # prepare_data uses them by default, and pooled rows enter the fit.
    prepared = cb.prepare_data(pooled, cb.ModelSpec(arm = "C"), fit_config = fit_config)
    assert prepared.silver_rates == one_table
    assert sum(len(v["y"]) for v in prepared.segments.values()) == len(pooled)

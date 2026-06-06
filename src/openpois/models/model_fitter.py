#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Fit POI-change models in JAX.

The likelihood is Bernoulli on a binary change indicator, with
P(change) = 1 - exp(-rate) derived from a Poisson event rate.
"""

import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import pandas as pd

from openpois.models.diagnostics import (
    flatten_chains,
    summarize_chain_draws,
    to_inference_data,
)
from openpois.models.jax_core import jax_rng, nuts_sample_multichain


def make_log_density(
    event_rate_fun: callable,
    param_likelihood: callable | None,
    data: dict[str, jnp.ndarray],
    target: jnp.ndarray,
    reduction_dtype: jnp.dtype | None = None,
    log_likelihood_fun: callable | None = None,
) -> callable:
    """
    Build a jitted log-density closure bound to fixed ``data`` / ``target``.

    The returned function takes only ``params`` as an argument, so BlackJAX
    traces a compact pytree (no ``ModelFitter`` instance). Captured arrays
    become XLA constants — the JIT cache key depends only on the shapes and
    dtypes of ``params``.

    By default, likelihood is the dense Bernoulli-on-Poisson form in stable
    shape:
        target=1: log(1 - exp(-rate)) via jnp.log(-jnp.expm1(-rate))
        target=0: -rate exactly.
    Models can short-circuit this by providing ``log_likelihood_fun(params,
    data, target)`` — e.g. the sufficient-statistics form used by
    ``RandomByTypeModel`` — and the dense per-observation path is skipped.

    Args:
        event_rate_fun, param_likelihood, data, target: as usual.
        reduction_dtype: If set, cast the per-observation log-probabilities to
            this dtype before the final ``sum`` reduction. For large N the
            fp32 reduction loses ~log10(N) significant digits; passing
            ``jnp.float64`` here recovers full precision. Requires
            ``jax_enable_x64`` to be True — call
            ``openpois.models.jax_core.enable_high_precision()`` first or the
            cast will be silently downcast to fp32 (we raise instead).
        log_likelihood_fun: Optional ``(params, data, target) -> scalar``
            callable that replaces the default dense log-likelihood. Used to
            plug in sufficient-statistics paths that avoid a reduction over
            all N observations.
    """
    if reduction_dtype is not None:
        reduction_dtype = jnp.dtype(reduction_dtype)
        needs_x64 = reduction_dtype == jnp.dtype(jnp.float64)
        if needs_x64 and not jax.config.read("jax_enable_x64"):
            raise ValueError(
                "reduction_dtype=jnp.float64 requires jax_enable_x64; call "
                "openpois.models.jax_core.enable_high_precision() before "
                "building the model."
            )

    @jax.jit
    def log_density(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        if log_likelihood_fun is not None:
            lp = log_likelihood_fun(params, data, target)
        else:
            rate = event_rate_fun(params, data)
            log_p = jnp.log(-jnp.expm1(-rate))
            log_1mp = -rate
            per_obs = target * log_p + (1.0 - target) * log_1mp
            if reduction_dtype is not None:
                per_obs = per_obs.astype(reduction_dtype)
            lp = jnp.sum(per_obs)
        if param_likelihood is not None:
            lp = lp + param_likelihood(params)
        return lp

    return log_density


class ModelFitter:
    """
    Fitter for POI change-rate models using BlackJAX NUTS.
    """

    EPSILON = 1e-6

    def __init__(
        self,
        event_rate_fun: callable,
        starting_params: dict[str, jnp.ndarray],
        data: dict[str, jnp.ndarray],
        target: jnp.ndarray,
        num_warmup: int | None = None,
        num_samples: int | None = None,
        num_draws: int | None = None,
        num_chains: int = 1,
        param_likelihood: callable | None = None,
        rng_key: jrd.KeyArray | None = None,
        verbose: bool = False,
        reduction_dtype: jnp.dtype | None = None,
        derive_draws: callable | None = None,
        log_likelihood_fun: callable | None = None,
        log_1md_fun: callable | None = None,
    ):
        """
        Args:
            event_rate_fun: Callable ``(params, data) -> rates`` returning a
                Poisson event rate per observation. ``data`` should bundle any
                covariates plus time bounds (e.g. ``t1``, ``t2``) the function
                needs.
            starting_params: Initial position for NUTS, as a pytree (dict) of
                arrays matching the signature expected by ``event_rate_fun``.
            data: Dict of arrays consumed by ``event_rate_fun``.
            target: Binary change indicator per observation.
            num_warmup: Number of NUTS warmup/adaptation steps. Defaults to
                ``num_draws`` if given (back-compat), otherwise 1 000.
            num_samples: Number of posterior draws retained. Defaults to
                ``num_draws`` if given (back-compat), otherwise 1 000.
            num_draws: **Deprecated** — legacy alias used for both
                ``num_warmup`` and ``num_samples`` when neither is provided.
            num_chains: Number of independent NUTS chains to run in parallel
                via ``jax.vmap``. ``num_chains > 1`` enables R-hat / cross-chain
                ESS diagnostics on ``self.diagnostics``. Default ``1``
                preserves legacy single-chain behaviour.
            param_likelihood: Optional ``(params) -> log_prior`` contribution
                added to the Bernoulli log-likelihood.
            rng_key: JAX PRNG key. Defaults to a fresh key from ``jax_rng()``.
            verbose: If True, emit a warmup progress bar (tqdm) plus a one-line
                diagnostic summary after sampling completes. When False, the
                summary only prints if any divergent transitions occurred.
            reduction_dtype: Optional dtype for the per-observation log-prob
                sum in the likelihood (e.g. ``jnp.float64``). ``None`` keeps
                fp32. See ``make_log_density`` for details.
            derive_draws: Optional ``(draws) -> draws'`` callable applied to
                the stacked posterior after ``fit()``. Used by reparameterised
                models (e.g. ``RandomByTypeModel`` under ``non_centered``) to
                expose natural-scale quantities. Raw sampler draws are kept on
                ``self.raw_param_draws`` regardless.
            log_likelihood_fun: Optional ``(params, data, target)`` callable
                that replaces the default dense per-observation likelihood.
                Used by models that expose a sufficient-statistics path
                (e.g. ``RandomByTypeModel`` with ``use_sufficient_stats=True``).
            log_1md_fun: Optional ``(params, data) -> log(1-δ)_per_row`` callable
                used by fresh-mode predictions when δ varies across observations
                (e.g. per-group in ``RandomByTypeModel``). If omitted, fresh mode
                falls back to a scalar ``params["logit_delta"]``.
        """
        if num_warmup is None:
            num_warmup = num_draws if num_draws is not None else 1_000
        if num_samples is None:
            num_samples = num_draws if num_draws is not None else 1_000
        self.event_rate_fun = event_rate_fun
        self.starting_params = starting_params
        self.data = data
        self.target = target
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.param_likelihood = param_likelihood
        self.rng_key = rng_key if rng_key is not None else jax_rng()
        self.verbose = verbose
        self.derive_draws = derive_draws
        self.log_1md_fun = log_1md_fun
        self.param_draws = None
        self.raw_param_draws = None
        self.chain_draws = None
        self.sampler_info = None
        self.warmup_params = None
        self.diagnostics = None
        self.model_finished = False
        self._log_density = make_log_density(
            event_rate_fun = event_rate_fun,
            param_likelihood = param_likelihood,
            data = data,
            target = target,
            reduction_dtype = reduction_dtype,
            log_likelihood_fun = log_likelihood_fun,
        )

    def calculate_probs(
        self,
        params: dict[str, jnp.ndarray],
        data: dict[str, jnp.ndarray],
        mode: str = "conditional",
    ) -> jnp.ndarray:
        """
        Map parameters + data to change probabilities in (EPSILON, 1 - EPSILON).

        Two regimes — see methodology §1.7 and §4.2 Step G:

        * ``"conditional"`` (default): P(change in next Δ | y_first = 0), i.e.
            ``1 − exp(−λ·Δ)``. δ drops out because conditioning on y_first=0
            assigns zero posterior to the δ-component. This is the right
            formula for rating an already-observed POI going forward.
        * ``"fresh"``: P(change by time Δ for a *fresh* individual),
            ``1 − (1−δ)·exp(−λ·Δ)``. Uses δ via
            ``log_sigmoid(-logit_delta)``. Requires ``"logit_delta"`` in
            ``params``; if absent, falls back to the conditional formula.
        """
        if mode not in ("conditional", "fresh"):
            raise ValueError(
                f"mode={mode!r} must be 'conditional' or 'fresh'"
            )
        change_rates = self.event_rate_fun(params, data)
        if mode == "fresh":
            if self.log_1md_fun is not None:
                log_1md = self.log_1md_fun(params, data)
                probs = 1.0 - jnp.exp(log_1md - change_rates)
            elif "logit_delta" in params:
                log_1md = jax.nn.log_sigmoid(-params["logit_delta"])
                probs = 1.0 - jnp.exp(log_1md - change_rates)
            else:
                probs = 1.0 - jnp.exp(-change_rates)
        else:
            probs = 1.0 - jnp.exp(-change_rates)
        return jnp.clip(probs, self.EPSILON, 1.0 - self.EPSILON)

    def calculate_lp(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Shim — delegates to the jitted log-density built in ``__init__``."""
        return self._log_density(params)

    def fit(self):
        """
        Draw posterior samples via BlackJAX NUTS with window adaptation.

        Stores:
          * ``self.param_draws`` — pytree, leading axis is (chains*draws) so
            existing consumers don't need to know chain count.
          * ``self.raw_param_draws`` — flattened draws before ``derive_draws``.
          * ``self.chain_draws`` — pytree with ``(num_chains, num_draws, ...)``
            leaves; used for R-hat / bulk ESS.
          * ``self.sampler_info`` — stacked NUTSInfo, leading ``(C, T)`` axes.
          * ``self.warmup_params`` — per-chain step size / inverse mass matrix.
          * ``self.diagnostics`` — R-hat / bulk-ESS DataFrame (only when
            ``num_chains > 1``; single-chain reports NaN R-hat).
        """
        chain_draws, self.sampler_info, self.warmup_params = (
            nuts_sample_multichain(
                log_density = self._log_density,
                init_position = self.starting_params,
                num_warmup = self.num_warmup,
                num_samples = self.num_samples,
                num_chains = self.num_chains,
                key = self.rng_key,
                verbose = self.verbose,
            )
        )
        self.chain_draws = chain_draws
        # Flatten (chains, draws) → (chains*draws) so get_parameter_table /
        # predict see a single leading draw axis regardless of chain count.
        flat_raw = flatten_chains(chain_draws)
        self.raw_param_draws = flat_raw
        if self.derive_draws is not None:
            self.param_draws = self.derive_draws(flat_raw)
        else:
            self.param_draws = flat_raw
        self.diagnostics = summarize_chain_draws(chain_draws)
        self.model_finished = True
        divergences = int(jnp.sum(self.sampler_info.is_divergent))
        if self.verbose or divergences > 0:
            self._print_sampler_summary(divergences = divergences)

    def _print_sampler_summary(self, divergences: int):
        """One-line NUTS diagnostic summary after ``fit()``."""
        info = self.sampler_info
        n_total = jax.tree_util.tree_leaves(self.param_draws)[0].shape[0]
        mean_accept = float(jnp.mean(info.acceptance_rate))
        mean_steps = float(jnp.mean(info.num_integration_steps))
        step_size_mean = float(jnp.mean(self.warmup_params["step_size"]))
        base = (
            f"NUTS: {self.num_chains} chains x {self.num_samples} draws | "
            f"accept={mean_accept:.3f} | "
            f"divergent={divergences}/{n_total} | "
            f"mean_steps={mean_steps:.1f} | step_size={step_size_mean:.4f}"
        )
        if self.num_chains > 1 and self.diagnostics is not None:
            rhat_max = float(self.diagnostics["rhat"].max())
            ess_min = float(self.diagnostics["ess_bulk"].min())
            base += f" | max_rhat={rhat_max:.3f} | min_ess={ess_min:.0f}"
        print(base)

    def get_parameter_draws(self) -> dict[str, jnp.ndarray]:
        """
        Return the posterior parameter draws as a pytree.

        Raises:
            ValueError: If ``fit()`` has not been run yet.
        """
        if self.param_draws is None:
            raise ValueError("Run fit() first")
        return self.param_draws

    def to_inference_data(self):
        """
        Return an ``arviz.InferenceData`` bundling posterior + sample_stats.

        Useful for ``az.plot_trace`` / ``az.summary`` / ``az.rhat`` in QA
        notebooks. Requires arviz to be installed.

        Raises:
            ValueError: If ``fit()`` has not been run.
            ImportError: If arviz is not installed.
        """
        if self.chain_draws is None:
            raise ValueError("Run fit() first")
        return to_inference_data(
            chain_draws = self.chain_draws,
            sampler_info = self.sampler_info,
        )

    def get_parameter_table(self, ui_width: float = 0.95) -> pd.DataFrame:
        """
        Summarize posterior draws as one row per scalar parameter.

        Vector/matrix parameters are unrolled with ``name[i]`` / ``name[i,j]``
        labels.

        Args:
            ui_width: Width of the uncertainty interval, in (0, 1).

        Returns:
            DataFrame with columns ``parameter``, ``mean``, ``lower``, ``upper``.
        """
        if self.param_draws is None:
            raise ValueError("Run fit() first")
        if not (0.0 < ui_width < 1.0):
            raise ValueError("ui_width must be between 0 and 1")
        lb = (1.0 - ui_width) / 2.0
        ub = 1.0 - lb

        rows = []
        for name, draws in self.param_draws.items():
            draws_np = np.asarray(draws)
            n_draws = draws_np.shape[0]
            flat = draws_np.reshape(n_draws, -1)
            param_shape = draws_np.shape[1:]
            for i in range(flat.shape[1]):
                if len(param_shape) == 0:
                    label = name
                else:
                    idx = np.unravel_index(i, param_shape)
                    label = f"{name}[{','.join(str(k) for k in idx)}]"
                col = flat[:, i]
                rows.append({
                    "parameter": label,
                    "mean": float(np.mean(col)),
                    "lower": float(np.quantile(col, lb, method = "linear")),
                    "upper": float(np.quantile(col, ub, method = "linear")),
                })
        return pd.DataFrame(rows)

    def predict(
        self,
        data: dict[str, jnp.ndarray] | None = None,
        ui_width: float = 0.95,
        mode: str = "conditional",
        chunk: int = 100_000,
    ) -> pd.DataFrame:
        """
        Posterior-predictive change probabilities.

        Vmaps ``calculate_probs`` over the stacked posterior draws in
        ``self.param_draws``, processing rows in blocks so the dense
        ``(draws, rows)`` probability matrix is never materialised in full
        (at national scale that matrix can be tens of GB).

        Args:
            data: Dict of arrays with the same keys as ``self.data``. Defaults
                to ``self.data``.
            ui_width: Width of the uncertainty interval, in (0, 1).
            mode: ``"conditional"`` (default) or ``"fresh"`` — see
                ``calculate_probs``. The conditional formula is δ-independent
                and is the right default for rating already-observed POIs.
            chunk: Number of rows scored per block. Peak memory scales with
                ``draws * chunk``.

        Returns:
            DataFrame with one row per observation and columns
            ``p_mean``, ``p_lower``, ``p_upper``.
        """
        if self.param_draws is None:
            raise ValueError("Run fit() first")
        if not (0.0 < ui_width < 1.0):
            raise ValueError("ui_width must be between 0 and 1")
        if data is None:
            data = self.data
        lb = (1.0 - ui_width) / 2.0
        ub = 1.0 - lb
        n = int(jnp.shape(data["dt"])[0])

        @jax.jit
        def block_stats(sub):
            all_probs = jax.vmap(
                lambda p: self.calculate_probs(p, sub, mode = mode)
            )(self.param_draws)
            return (
                jnp.mean(all_probs, axis = 0),
                jnp.quantile(all_probs, lb, axis = 0, method = "linear"),
                jnp.quantile(all_probs, ub, axis = 0, method = "linear"),
            )

        means, lowers, uppers = [], [], []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            # Slice only the per-row arrays (length n); leave any scalars/other.
            sub = {
                k: (v[start:end] if getattr(v, "shape", None) and v.shape[0] == n
                    else v)
                for k, v in data.items()
            }
            m, lo, hi = block_stats(sub)
            means.append(np.asarray(m))
            lowers.append(np.asarray(lo))
            uppers.append(np.asarray(hi))
        return pd.DataFrame({
            "p_mean": np.concatenate(means) if means else np.array([]),
            "p_lower": np.concatenate(lowers) if lowers else np.array([]),
            "p_upper": np.concatenate(uppers) if uppers else np.array([]),
        })

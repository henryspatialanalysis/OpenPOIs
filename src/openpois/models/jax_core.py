#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Core functions for fitting Bayesian hierarchical models using JAX.
"""

import os

from numpy.random import randint

import jax
import jax.numpy as jnp
import jax.random as jrd
import blackjax


# Probe with a real import, not just find_spec: fastprogress can be installed
# yet unusable (it imports IPython, which may be absent), in which case
# BlackJAX's progress-bar callback raises at run time. Treat a broken
# fastprogress as absent so verbose fits silently skip the progress bar.
try:
    import fastprogress  # noqa: F401

    _HAS_FASTPROGRESS = True
except Exception:
    _HAS_FASTPROGRESS = False


def _configure_compilation_cache():
    """
    Point JAX's persistent-compilation cache at a stable user-local directory.

    XLA compiles of our jitted log-density and the BlackJAX NUTS scan cost
    several seconds per shape/dtype combination; caching them across runs
    makes iteration noticeably faster. Cache key depends on HLO only, so
    stale entries fall out naturally when model shapes change.
    The env var ``OPENPOIS_JAX_CACHE_DIR`` overrides the default path for
    CI or shared-runner scenarios.
    """
    cache_dir = os.environ.get(
        "OPENPOIS_JAX_CACHE_DIR",
        os.path.expanduser("~/.cache/openpois/jax"),
    )
    os.makedirs(cache_dir, exist_ok = True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)


_configure_compilation_cache()


def enable_high_precision():
    """
    Enable JAX's fp64 mode (``jax_enable_x64``) process-wide.

    Required when passing ``reduction_dtype=jnp.float64`` to
    ``ModelFitter`` — without the flag, JAX silently downcasts all fp64
    values to fp32. Call this before constructing any traced arrays.

    Cost: fp64 arithmetic roughly doubles memory for any array that's
    actually in fp64. Gradient compute over an fp32 data array plus an
    fp64 reduction is a good trade-off at N > ~1e5.
    """
    jax.config.update("jax_enable_x64", True)


def jax_rng():
    """
    Generate a JAX random key.
    """
    rng = jrd.PRNGKey(randint(2**32 - 1))
    return rng


def random_markov_chain(
    kernel: callable,
    init_state: blackjax.State,
    num_draws: int = 1_000,
    key: jrd.KeyArray | None = None,
) -> tuple:
    """
    Run a BlackJAX kernel for ``num_draws`` steps via ``jax.lax.scan``.

    Returns both the trajectory of states and the per-step info pytree (for
    NUTS: ``is_divergent``, ``acceptance_rate``, ``num_integration_steps``,
    ``energy`` …), so callers can compute sampler diagnostics.

    Args:
        kernel: BlackJAX kernel step function.
        init_state: Initial sampler state.
        num_draws: Number of draws to generate.
        key: JAX random key.

    Returns:
        ``(states, infos)`` — both stacked along a leading ``num_draws`` axis.
    """
    if key is None:
        key = jax_rng()

    @jax.jit
    def one_step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)

    keys = jrd.split(key = key, num = num_draws)
    _, (states, infos) = jax.lax.scan(f = one_step, init = init_state, xs = keys)
    return states, infos


def _nuts_sample_core(
    log_density: callable,
    init_position,
    num_warmup: int,
    num_samples: int,
    key: jrd.KeyArray,
    show_progress: bool = False,
) -> tuple:
    """
    Pure sampling body for one NUTS chain: warmup + scan. No prints.

    Returns ``(states.position, sampler_info, warmup_params)``. Factored out
    of ``nuts_sample`` so it can be ``vmap``ped for multi-chain runs.
    """
    warmup_key, sample_key = jrd.split(key = key, num = 2)
    warmup = blackjax.window_adaptation(
        algorithm = blackjax.nuts,
        logdensity_fn = log_density,
        progress_bar = show_progress,
    )
    (state, warmup_params), _ = warmup.run(
        rng_key = warmup_key,
        position = init_position,
        num_steps = num_warmup,
    )
    kernel = blackjax.nuts(
        logdensity_fn = log_density,
        **warmup_params,
    ).step
    states, sampler_info = random_markov_chain(
        key = sample_key,
        kernel = kernel,
        init_state = state,
        num_draws = num_samples,
    )
    return states.position, sampler_info, warmup_params


def nuts_sample(
    log_density: callable,
    init_position: blackjax.State,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    key: jrd.KeyArray | None = None,
    verbose: bool = False,
) -> tuple:
    """
    Draw posterior samples via BlackJAX NUTS + window adaptation (1 chain).

    Args:
        log_density: Log-density function ``params -> scalar``.
        init_position: Initial parameter pytree.
        num_warmup: Number of warmup/adaptation steps. Window adaptation tunes
            step size + (diagonal) inverse mass matrix over these iterations.
        num_samples: Number of posterior draws retained after warmup.
        key: JAX random key.
        verbose: If True, emit a progress bar during warmup (BlackJAX built-in)
            and print a short banner before the sampling scan begins. Sampling
            itself runs as a single ``jax.lax.scan`` so it stays silent — the
            summary is printed by the caller after ``fit()``.

    Returns:
        ``(param_draws, sampler_info, warmup_params)`` — posterior draws as a
        pytree matching ``init_position`` (leading draw axis), the stacked
        NUTSInfo from sampling, and the adapted ``(step_size,
        inverse_mass_matrix)`` dict from warmup.
    """
    if key is None:
        key = jax_rng()
    show_progress = verbose and _HAS_FASTPROGRESS
    if verbose:
        note = "" if show_progress else " (install fastprogress for a progress bar)"
        print(f"NUTS warmup: {num_warmup} steps (window adaptation){note}...")
    position, sampler_info, warmup_params = _nuts_sample_core(
        log_density = log_density,
        init_position = init_position,
        num_warmup = num_warmup,
        num_samples = num_samples,
        key = key,
        show_progress = show_progress,
    )
    if verbose:
        step_size = float(warmup_params["step_size"])
        print(
            f"NUTS warmup done | step_size={step_size:.4f} | "
            f"sampled {num_samples} draws."
        )
    return position, sampler_info, warmup_params


def _jitter_init_position(
    init_position,
    num_chains: int,
    jitter: float,
    key: jrd.KeyArray,
):
    """Stack ``init_position`` ``num_chains`` ways and add i.i.d. N(0, jitter) noise."""
    leaves, treedef = jax.tree_util.tree_flatten(init_position)
    leaf_keys = jrd.split(key = key, num = len(leaves))
    stacked = []
    for leaf, leaf_key in zip(leaves, leaf_keys):
        chain_keys = jrd.split(key = leaf_key, num = num_chains)
        shape = jnp.shape(leaf)
        broadcast = jnp.broadcast_to(leaf, (num_chains,) + shape)
        noise = jax.vmap(
            lambda k: jrd.normal(k, shape) * jitter
        )(chain_keys)
        stacked.append(broadcast + noise)
    return jax.tree_util.tree_unflatten(treedef, stacked)


def nuts_sample_multichain(
    log_density: callable,
    init_position,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 4,
    key: jrd.KeyArray | None = None,
    init_jitter: float = 0.05,
    verbose: bool = False,
) -> tuple:
    """
    Draw posterior samples via NUTS on ``num_chains`` chains in parallel.

    Each chain gets its own window-adapted step size + mass matrix and its
    own jittered starting position (i.i.d. N(0, ``init_jitter``) added to
    ``init_position``). Chains are vmapped so they share one XLA compile.

    Returns:
        ``(chain_draws, chain_info, chain_warmup)`` — same shapes as
        ``nuts_sample`` but with a leading ``num_chains`` axis on every leaf.
    """
    if num_chains < 1:
        raise ValueError(f"num_chains must be >= 1, got {num_chains}")
    if key is None:
        key = jax_rng()
    if num_chains == 1:
        # Preserve the single-chain banner/progress-bar UX and avoid vmap
        # overhead when the user hasn't asked for multi-chain sampling.
        draws, info, warmup = nuts_sample(
            log_density = log_density,
            init_position = init_position,
            num_warmup = num_warmup,
            num_samples = num_samples,
            key = key,
            verbose = verbose,
        )
        # Add a leading singleton chain axis so downstream code is shape-uniform.

        def add_axis(x):
            return jnp.expand_dims(x, axis = 0)

        return (
            jax.tree_util.tree_map(add_axis, draws),
            jax.tree_util.tree_map(add_axis, info),
            jax.tree_util.tree_map(add_axis, warmup),
        )

    jitter_key, chain_seed = jrd.split(key = key, num = 2)
    chain_keys = jrd.split(key = chain_seed, num = num_chains)
    stacked_init = _jitter_init_position(
        init_position = init_position,
        num_chains = num_chains,
        jitter = init_jitter,
        key = jitter_key,
    )
    if verbose:
        print(
            f"NUTS: {num_chains} chains | {num_warmup} warmup + "
            f"{num_samples} samples | jitter={init_jitter}"
        )

    def _run_chain(k, init):
        return _nuts_sample_core(
            log_density = log_density,
            init_position = init,
            num_warmup = num_warmup,
            num_samples = num_samples,
            key = k,
            show_progress = False,
        )

    return jax.vmap(_run_chain)(chain_keys, stacked_init)


def generate_predictive_draws(
    posterior_predictive: callable,
    param_draws: dict[str, jnp.ndarray],
    num_draws: int = 1_000,
    key: jrd.KeyArray | None = None,
) -> jnp.ndarray:
    """
    Generate predictive draws from a posterior distribution using JAX's vmap.

    Args:
        posterior_predictive: Posterior predictive function (data already incorporated)
        param_draws: Parameter draws.
        num_draws: Number of draws to generate.
        key: JAX random key.
    """
    if key is None:
        key = jax_rng()
    keys = jrd.split(key = key, num = num_draws)
    pred_draws = jax.vmap(
        fun = posterior_predictive,
        in_axes = (0, 0),
    )(keys, param_draws)
    return pred_draws.T

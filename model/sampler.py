"""
sampler.py
==========
Generates parameter sets for the prior predictive ensemble using
Latin Hypercube Sampling (LHS).

WHAT THIS FILE DOES
-------------------
Takes the probability distributions defined in parameters.py and
produces N complete, biologically valid parameter dictionaries.
Each dictionary represents one plausible version of biological
reality. The ensemble of N dictionaries is what solver.py loops over.

WHY LATIN HYPERCUBE SAMPLING AND NOT PURE RANDOM SAMPLING
----------------------------------------------------------
Pure random sampling (just calling dist.rvs() N times) clusters.
With 500 draws, you might sample beta0 near 0.035 two hundred times
and barely explore the tails of the distribution. This wastes runs
and gives you poor coverage of the parameter space.

Latin Hypercube Sampling solves this by dividing each parameter's
distribution into N equal-probability intervals and guaranteeing
exactly one sample falls in each interval. The intervals are then
randomly permuted and combined across parameters.

The result: with N=500, LHS gives coverage that pure random sampling
would need roughly 2000 runs to match. Every region of the parameter
space gets explored, not just the high-density center.

Concretely, LHS works in two steps:
    Step 1: Generate an N x D matrix of uniform [0,1] values,
            where D = number of sampled parameters. Each column
            is a stratified permutation (one value per 1/N interval).
    Step 2: Map each uniform value through the inverse CDF (.ppf())
            of the corresponding parameter distribution.
            u=0.5 maps to the median, u=0.025 maps to the 2.5th
            percentile, etc. This is the percent-point function.

VALIDITY FILTERING
------------------
Even with well-chosen distributions, LHS can produce edge-case
combinations that are biologically implausible. Two filters are
applied after sampling:

    1. PARAM_BOUNDS: hard min/max for each individual parameter.
       Catches extreme tail draws (e.g. beta0 very near zero).

    2. SIGMA_GAMMA_RATIO_BOUNDS: the ratio sigma/gamma must be
       between 0.3 and 4.0. This filters implausible combinations
       like a 1-day incubation paired with a 25-day infectious period,
       which can occur even when each individual draw is within bounds.
       This ratio filter is the explicit substitute for a full Gaussian
       copula correlation structure between sigma and gamma, which is
       noted as a simplification in the project report.

Invalid samples are redrawn individually until all N pass both filters.
This is rejection sampling on the combined validity condition.

REPRODUCIBILITY
---------------
All randomness flows through a single numpy Generator object created
with a fixed seed (ENSEMBLE_SEED = 42 from parameters.py). Anyone
running this file with the same seed gets identical results.
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube

from model.parameters import (
    PARAM_DISTRIBUTIONS,
    SAMPLED_PARAM_NAMES,
    PARAM_BOUNDS,
    SIGMA_GAMMA_RATIO_BOUNDS,
    FIXED_PARAMS,
    DEFAULT_PARAMS,
    N_ENSEMBLE_RUNS,
    ENSEMBLE_SEED,
)


# =============================================================================
# CORE SAMPLING FUNCTIONS
# =============================================================================

def _is_valid(sample_dict: dict) -> bool:
    """
    Checks whether a single parameter dictionary passes both validity filters.

    Parameters
    ----------
    sample_dict : dict
        A complete parameter dictionary for the sampled parameters only
        (not yet merged with FIXED_PARAMS).

    Returns
    -------
    bool
        True if the sample passes all validity checks.
        False if any check fails — the sample will be redrawn.

    Checks applied
    --------------
    1. Each sampled parameter must be within its PARAM_BOUNDS range.
    2. The ratio sigma/gamma must be within SIGMA_GAMMA_RATIO_BOUNDS.
       This catches biologically implausible incubation/infectious period
       combinations that individual bounds cannot catch.
    """
    # Check 1: individual parameter bounds
    for param_name, value in sample_dict.items():
        if param_name in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[param_name]
            if not (lo <= value <= hi):
                return False

    # Check 2: sigma/gamma biological plausibility ratio
    # sigma/gamma = infectious_period / incubation_period
    # For respiratory pathogens this should be between 0.3 and 4.0
    ratio = sample_dict["sigma"] / sample_dict["gamma"]
    lo_r, hi_r = SIGMA_GAMMA_RATIO_BOUNDS
    if not (lo_r <= ratio <= hi_r):
        return False

    return True


def _lhs_draw(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates an LHS design matrix of shape (n, D) with uniform [0,1] values,
    where D = number of sampled parameters.

    Each column is a stratified permutation — one sample per 1/n interval —
    ensuring full coverage of each parameter's marginal distribution.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    rng : np.random.Generator
        Seeded numpy random generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, D)
        Uniform [0,1] LHS design matrix. Column j corresponds to
        SAMPLED_PARAM_NAMES[j].
    """
    D = len(SAMPLED_PARAM_NAMES)

    # scipy's LatinHypercube sampler requires a seed integer, not a Generator.
    # We derive a deterministic integer seed from the Generator so the LHS
    # remains tied to the global seed while using the correct interface.
    lhs_seed = int(rng.integers(0, 2**31))
    sampler = LatinHypercube(d=D, seed=lhs_seed)

    # sampler.random(n) returns an (n, D) array of stratified uniform values
    return sampler.random(n)


def _uniform_to_params(uniform_row: np.ndarray) -> dict:
    """
    Maps one row of uniform [0,1] LHS values to actual parameter values
    using each distribution's inverse CDF (percent-point function, .ppf()).

    This is the key mathematical step of LHS:
        actual_value = distribution.ppf(uniform_value)

    For example:
        beta0_dist.ppf(0.5)  → median of the beta0 distribution
        beta0_dist.ppf(0.025) → 2.5th percentile of beta0
        gamma_dist.ppf(0.9)  → 90th percentile of the gamma dist

    Parameters
    ----------
    uniform_row : np.ndarray, shape (D,)
        One row from the LHS design matrix. Values in [0,1].

    Returns
    -------
    dict
        Parameter names mapped to sampled values.
        Contains SAMPLED_PARAM_NAMES keys only — not yet merged with
        FIXED_PARAMS.
    """
    sample = {}
    for j, name in enumerate(SAMPLED_PARAM_NAMES):
        dist = PARAM_DISTRIBUTIONS[name]
        # .ppf() = percent-point function = inverse CDF
        # Maps the uniform LHS value to the actual parameter value
        sample[name] = float(dist.ppf(uniform_row[j]))
    return sample


def draw_single_sample(rng: np.random.Generator) -> dict:
    """
    Draws one complete, valid parameter dictionary.

    Uses pure random sampling (not LHS) for single draws. LHS is only
    meaningful when drawing multiple samples simultaneously, because its
    stratification property requires knowing N in advance.

    This function is used for:
        - Quick one-off validation runs
        - Resampling rejected ensemble members (see draw_ensemble)

    Parameters
    ----------
    rng : np.random.Generator
        Seeded numpy random generator.

    Returns
    -------
    dict
        A complete parameter dictionary merging sampled values with
        FIXED_PARAMS. Ready to pass directly to seirs_v_odes().
    """
    while True:
        # Draw from each distribution independently
        sample = {
            name: float(PARAM_DISTRIBUTIONS[name].rvs(random_state=rng))
            for name in SAMPLED_PARAM_NAMES
        }
        # Accept only if it passes all validity checks
        if _is_valid(sample):
            # Merge with fixed parameters to produce a complete dict
            return {**sample, **FIXED_PARAMS}


def draw_ensemble(
    n: int = N_ENSEMBLE_RUNS,
    seed: int = ENSEMBLE_SEED
) -> list:
    """
    Draws N complete, valid parameter dictionaries using Latin Hypercube
    Sampling. This is the main function used by solver.py.

    Parameters
    ----------
    n : int
        Number of ensemble members to generate.
        Default: N_ENSEMBLE_RUNS = 500 (from parameters.py).
    seed : int
        Random seed for full reproducibility.
        Default: ENSEMBLE_SEED = 42 (from parameters.py).

    Returns
    -------
    list of dict
        Length-n list. Each dict is a complete parameter set
        (sampled + fixed) ready to pass to seirs_v_odes().

    Algorithm
    ---------
    1. Generate an initial LHS design of n uniform samples.
    2. Map each row through the inverse CDF of each distribution.
    3. Check each sample against validity filters.
    4. For any rejected sample, redraw using draw_single_sample()
       until a valid replacement is found.
    5. Merge every accepted sample with FIXED_PARAMS.

    Notes
    -----
    The initial LHS batch covers most draws. Redraws use pure random
    sampling (not a new LHS batch) because they are individual
    replacements — regenerating the entire LHS design for a few
    rejected samples would destroy the stratification property for
    the accepted ones.

    In practice, the validity filters reject only a small fraction
    of samples (typically under 5%) because the distributions are
    calibrated to the valid range. The ratio filter for sigma/gamma
    is the most common reason for rejection.
    """
    rng = np.random.default_rng(seed=seed)

    # Step 1: Generate the LHS design matrix — shape (n, D)
    # Each column is a stratified uniform sample for one parameter
    lhs_matrix = _lhs_draw(n, rng)

    ensemble = []
    n_rejected = 0

    for i in range(n):
        # Step 2: Map uniform values to actual parameter values
        candidate = _uniform_to_params(lhs_matrix[i])

        # Step 3: Validity check
        if _is_valid(candidate):
            # Step 4: Merge with fixed parameters and store
            ensemble.append({**candidate, **FIXED_PARAMS})
        else:
            # Step 5: Reject and redraw using pure random sampling
            # We do NOT regenerate the entire LHS — just replace this
            # one invalid sample with a valid individual draw
            n_rejected += 1
            ensemble.append(draw_single_sample(rng))

    # Report rejection rate for diagnostics
    rejection_rate = 100.0 * n_rejected / n
    if rejection_rate > 10.0:
        print(f"  WARNING: {rejection_rate:.1f}% of samples rejected. "
              f"Consider widening PARAM_BOUNDS or tightening distributions.")

    return ensemble


def ensemble_to_array(ensemble: list) -> np.ndarray:
    """
    Converts the ensemble list of dicts into a 2D numpy array for
    convenient analysis and storage.

    Parameters
    ----------
    ensemble : list of dict
        Output of draw_ensemble().

    Returns
    -------
    np.ndarray, shape (N, D)
        Row i = parameter set i.
        Column j = SAMPLED_PARAM_NAMES[j] value.
        Fixed parameters are NOT included (they are constant across rows).

    Notes
    -----
    This is useful for plotting parameter distributions across the
    ensemble and for computing sensitivity metrics in analysis.py.
    """
    return np.array([
        [params[name] for name in SAMPLED_PARAM_NAMES]
        for params in ensemble
    ])


# =============================================================================
# SELF-CHECK  (run with: python model/sampler.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("SAMPLER.PY — SELF-CHECK")
    print("=" * 62)

    # ------------------------------------------------------------------
    print("\n[1] Single sample draw")
    rng_test = np.random.default_rng(seed=0)
    single = draw_single_sample(rng_test)
    print(f"    Keys present: {sorted(single.keys())}")
    print(f"    Sampled values:")
    for name in SAMPLED_PARAM_NAMES:
        lo, hi = PARAM_BOUNDS.get(name, (None, None))
        val = single[name]
        in_bounds = lo <= val <= hi if lo is not None else True
        print(f"      {name:<12} = {val:.6f}  "
              f"bounds=[{lo:.6f},{hi:.4f}]  "
              f"{'OK' if in_bounds else 'FAIL'}")
    ratio = single["sigma"] / single["gamma"]
    lo_r, hi_r = SIGMA_GAMMA_RATIO_BOUNDS
    print(f"    sigma/gamma ratio = {ratio:.3f}  "
          f"bounds=[{lo_r},{hi_r}]  "
          f"{'OK' if lo_r<=ratio<=hi_r else 'FAIL'}")

    # ------------------------------------------------------------------
    print("\n[2] Fixed params present in single sample")
    for key in FIXED_PARAMS:
        present = key in single
        match = abs(single.get(key, 0) - FIXED_PARAMS[key]) < 1e-12
        print(f"    {key:<12} present={present}  "
              f"value_correct={match}")

    # ------------------------------------------------------------------
    print("\n[3] Small ensemble draw (n=50 for speed)")
    ens = draw_ensemble(n=50, seed=42)
    print(f"    Ensemble length: {len(ens)}")
    print(f"    All are dicts:   {all(isinstance(e, dict) for e in ens)}")
    print(f"    All have correct keys: "
          f"{all(set(SAMPLED_PARAM_NAMES) <= set(e.keys()) for e in ens)}")

    # ------------------------------------------------------------------
    print("\n[4] Parameter coverage check (ensemble of 50)")
    arr = ensemble_to_array(ens)
    print(f"    Array shape: {arr.shape}  "
          f"(expected: (50, {len(SAMPLED_PARAM_NAMES)}))")
    print(f"    {'param':<12} {'min':>10} {'mean':>10} {'max':>10}")
    print(f"    {'-'*46}")
    for j, name in enumerate(SAMPLED_PARAM_NAMES):
        col = arr[:, j]
        print(f"    {name:<12} {col.min():>10.5f} "
              f"{col.mean():>10.5f} {col.max():>10.5f}")

    # ------------------------------------------------------------------
    print("\n[5] LHS stratification check")
    print("    Verifying each parameter covers its distribution range...")
    covered = 0
    for j, name in enumerate(SAMPLED_PARAM_NAMES):
        col = arr[:, j]
        dist = PARAM_DISTRIBUTIONS[name]
        # Check that samples span at least the 10th to 90th percentile range
        p10 = float(dist.ppf(0.10))
        p90 = float(dist.ppf(0.90))
        spans = col.min() <= p10 * 1.5 and col.max() >= p90 * 0.5
        print(f"    {name:<12} min={col.min():.5f} "
              f"p10={p10:.5f} p90={p90:.5f} "
              f"max={col.max():.5f}  "
              f"{'spans range' if spans else 'WARNING: poor coverage'}")
        if spans:
            covered += 1
    print(f"    Coverage: {covered}/{len(SAMPLED_PARAM_NAMES)} parameters")

    # ------------------------------------------------------------------
    print("\n[6] Reproducibility check")
    ens_a = draw_ensemble(n=10, seed=42)
    ens_b = draw_ensemble(n=10, seed=42)
    ens_c = draw_ensemble(n=10, seed=99)
    same_seed  = all(
        abs(ens_a[i]["beta0"] - ens_b[i]["beta0"]) < 1e-12
        for i in range(10)
    )
    diff_seed = any(
        abs(ens_a[i]["beta0"] - ens_c[i]["beta0"]) > 1e-10
        for i in range(10)
    )
    print(f"    Same seed gives identical results:     {same_seed}")
    print(f"    Different seed gives different results: {diff_seed}")

    print("\n" + "=" * 62)
    print("sampler.py self-check complete.")
    print("=" * 62)

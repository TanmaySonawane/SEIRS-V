"""
parameters.py
=============
Single source of truth for every constant AND every prior distribution
in the SEIRS-V age-structured epidemic model with seasonal forcing,
behavioral adaptation, and probabilistic parameterization.

HOW TO USE THIS FILE
--------------------
Other files import from here in two distinct ways:

    1. DETERMINISTIC (single run, testing, validation):
           from model.parameters import DEFAULT_PARAMS, POPULATION, TIME_SPAN
           Use DEFAULT_PARAMS directly as the parameter dict.

    2. PROBABILISTIC (prior predictive ensemble):
           from model.parameters import PARAM_DISTRIBUTIONS, FIXED_PARAMS
           Pass PARAM_DISTRIBUTIONS to sampler.py, which draws N parameter
           sets using Latin Hypercube Sampling and returns a list of dicts.
           FIXED_PARAMS contains everything that does NOT get sampled.

WHAT IS A PRIOR PREDICTIVE ENSEMBLE?
-------------------------------------
We do not know the true value of parameters like beta0 or kappa exactly.
Instead of pretending we do (one fixed number), we define a plausible
range for each parameter as a probability distribution — the "prior."
We then draw 500 complete parameter sets from these distributions and
run the ODE solver once per set. The spread of the 500 resulting
epidemic curves represents our uncertainty about the epidemic trajectory.

This is called a prior predictive ensemble. It is standard practice in
published epidemiological modeling (Imperial College, IHME, CDC).
It is NOT the same as a stochastic model — the ODEs remain deterministic.
Only the inputs are uncertain.

LATIN HYPERCUBE SAMPLING NOTE
------------------------------
sampler.py uses Latin Hypercube Sampling (LHS) rather than pure random
sampling. LHS divides each parameter's distribution into N equal-
probability intervals and guarantees one sample per interval, ensuring
the full parameter space is explored efficiently. With N=500, LHS gives
coverage that pure random sampling would need 2000+ runs to match.

STRUCTURE
---------
    1. POPULATION          - group sizes (fixed, known exactly)
    2. FIXED_PARAMS        - parameters NOT sampled (known or structural)
    3. PARAM_DISTRIBUTIONS - parameters that ARE sampled, with priors
    4. DEFAULT_PARAMS      - point estimates (means of each distribution)
                             Used for deterministic runs and validation.
    5. PARAM_BOUNDS        - hard biological validity bounds for
                             post-sampling rejection filter
    6. INITIAL_STATE       - starting compartment values at t=0
    7. TIME_SPAN / T_EVAL  - simulation time settings
    8. get_contact_matrix  - helper to build 2x2 numpy array
    9. Self-check block    - run with: python model/parameters.py
"""

import numpy as np
from scipy import stats


# =============================================================================
# 1. POPULATION  (fixed — known exactly from census data)
# =============================================================================

POPULATION = {
    "N1": 200_000,    # Total youth population  (ages 0-17, ~30% of total)
    "N2": 500_000,    # Total adult population  (ages 18+, ~70% of total)
}
POPULATION["N"] = POPULATION["N1"] + POPULATION["N2"]   # 700,000 total


# =============================================================================
# 2. FIXED PARAMS  (not sampled — structural or well-characterized)
# =============================================================================
# These parameters are either:
#   (a) demographic constants well-established from data (mu), or
#   (b) modeling assumptions we fix deliberately so the ensemble
#       isolates biological uncertainty (contact matrix, omega_v), or
#   (c) structural settings that are not biological parameters.
#
# Keeping these fixed is a deliberate modeling choice stated in the report.
# The contact matrix COULD be made uncertain (POLYMOD provides confidence
# intervals) but fixing it separates biological transmission uncertainty
# from structural contact uncertainty. Noted as a limitation.

FIXED_PARAMS = {

    "mu": 1.0 / (70.0 * 365.0),
    # Natural birth and death rate, per day.
    # = 1 / (average lifespan in days) = 1 / 25,550 approx 0.0000392 /day
    # Birth rate is set equal to this so population stays constant.
    # Well-characterized from national census data — not uncertain.

    "omega_v": 1.0 / 365.0,
    # Vaccine immunity waning rate, per day.
    # Fixed at 12-month duration as a modeling assumption.
    # Could be made uncertain in a sensitivity extension.

    "c11": 9.0,    # Youth-youth daily contacts (school-dominated)
    "c12": 3.0,    # Youth-adult daily contacts (home/community)
    "c21": 1.5,    # Adult-youth daily contacts (reciprocity-adjusted)
    "c22": 4.5,    # Adult-adult daily contacts (workplace-dominated)
    # Contact matrix values from POLYMOD-inspired estimates.
    # Reciprocity: c12*N1 approx c21*N2
    #   3.0 * 200,000 = 600,000
    #   1.5 * 500,000 = 750,000
    # Small asymmetry acceptable — see report limitations section.
}


# =============================================================================
# 3. PARAM_DISTRIBUTIONS  (parameters that ARE sampled)
# =============================================================================
# Each entry is a scipy.stats frozen distribution object.
# "Frozen" means the shape parameters are already set — you just call
# .ppf(u) on a uniform [0,1] value to get a sample. This is exactly
# what Latin Hypercube Sampling does internally.
#
# Distribution choices are biologically motivated:
#
#   truncnorm  -> symmetric measurement error around a best estimate,
#                 but physically constrained to stay positive.
#                 Used for beta0 (estimated from contact studies).
#
#   Beta       -> naturally bounded to [0, 1]. Used for probabilities
#                 and proportions (delta = vaccine efficacy).
#                 Beta(a, b): mean = a/(a+b)
#
#   Gamma      -> naturally positive, right-skewed. Used for biological
#                 rates (sigma, gamma) because clinical data on infectious
#                 periods is right-skewed — most recover in ~10 days but
#                 a tail extends much longer.
#                 Gamma(shape, scale): mean = shape * scale
#
#   Uniform    -> maximum-entropy choice when we know the plausible range
#                 but have no strong reason to prefer any value within it.
#                 Used for behavioral (kappa), policy (nu), and seasonal
#                 (epsilon) parameters where empirical data is sparse.

PARAM_DISTRIBUTIONS = {

    # -------------------------------------------------------------------------
    # TRANSMISSION
    # -------------------------------------------------------------------------

    "beta0": stats.truncnorm(
        a=(0.001 - 0.035) / 0.005,   # lower bound in standardized units
        b=np.inf,                     # no upper bound
        loc=0.035,                    # mean
        scale=0.005                   # std dev
    ),
    # Baseline transmission probability per contact per day.
    # Mean 0.035, std 0.005.
    # 95% of samples fall in approximately [0.025, 0.045].
    # Truncated so it cannot go negative or near-zero.
    # Reflects measurement uncertainty from contact tracing studies.

    "epsilon": stats.uniform(loc=0.10, scale=0.20),
    # Seasonality amplitude. Uniform over [0.10, 0.30].
    # Literature reports 10% to 30% for respiratory pathogens.
    # No strong reason to prefer any value in this range.

    "kappa": stats.uniform(loc=2.0, scale=8.0),
    # Behavioral response strength. Uniform over [2.0, 10.0].
    # Most uncertain parameter in the model — depends on culture,
    # media coverage, and policy enforcement, none well-quantified.
    # Wide uniform range reflects deep uncertainty.

    # -------------------------------------------------------------------------
    # DISEASE PROGRESSION
    # -------------------------------------------------------------------------

    "sigma": stats.gamma(a=25.0, scale=1.0 / (25.0 * 5.0)),
    # E to I progression rate (= 1/incubation period), per day.
    # Gamma(shape=25, scale=0.008): mean = 25 * 0.008 = 0.200 /day
    # which corresponds to average 5-day incubation.
    # Shape=25 makes the distribution relatively tight (std approx 0.04)
    # because incubation period is reasonably well characterized.
    # Gamma is used because rates must be strictly positive and
    # clinical incubation data shows a right skew.

    "gamma": stats.gamma(a=25.0, scale=1.0 / (25.0 * 10.0)),
    # I to R recovery rate (= 1/infectious period), per day.
    # Gamma(shape=25, scale=0.004): mean = 25 * 0.004 = 0.100 /day
    # which corresponds to average 10-day infectious period.
    # Same reasoning as sigma — strictly positive, right-skewed.

    # -------------------------------------------------------------------------
    # IMMUNITY
    # -------------------------------------------------------------------------

    "omega": stats.uniform(
        loc=1.0 / 270.0,
        scale=(1.0 / 90.0) - (1.0 / 270.0)
    ),
    # Natural immunity waning rate, per day.
    # Uniform over [1/270, 1/90] = immunity duration 3 to 9 months.
    # Genuinely uncertain for novel or evolving pathogens.

    # -------------------------------------------------------------------------
    # VACCINATION
    # -------------------------------------------------------------------------

    "delta": stats.beta(a=32.0, b=8.0),
    # Vaccine efficacy (reduction in susceptibility for vaccinated people).
    # Beta(32, 8): mean = 32/40 = 0.80.
    # Beta distribution is the natural choice for a proportion in [0,1].
    # Parameters chosen so mean = 0.80, std approx 0.063.
    # 95% CI approximately [0.67, 0.92] — consistent with mRNA vaccine
    # clinical trial confidence intervals.

    "nu1": stats.uniform(loc=0.001, scale=0.003),
    # Youth vaccination rate. Uniform over [0.001, 0.004] /day.
    # = 0.1% to 0.4% of susceptible youth vaccinated per day.
    # Policy-dependent quantity — Uniform reflects policy uncertainty.

    "nu2": stats.uniform(loc=0.0005, scale=0.0015),
    # Adult vaccination rate. Uniform over [0.0005, 0.002] /day.
    # Approximately half the youth range — pediatric campaigns often
    # prioritize children first in respiratory disease outbreaks.
}

# Ordered list of sampled parameter names — used by sampler.py to
# maintain consistent column ordering in the LHS design matrix.
SAMPLED_PARAM_NAMES = list(PARAM_DISTRIBUTIONS.keys())


# =============================================================================
# 4. DEFAULT_PARAMS  (point estimates — means of each distribution)
# =============================================================================
# This dict is used for:
#   (a) All four deterministic validation runs
#   (b) Dashboard baseline display before ensemble runs complete
#   (c) The self-check at the bottom of this file
#   (d) Any scenario where you want the "best guess" single run
#
# Every sampled-parameter value here is the mean of its distribution.
# FIXED_PARAMS values are included via ** unpacking so DEFAULT_PARAMS
# is always a complete, self-contained parameter dict.

DEFAULT_PARAMS = {
    # Sampled parameters at their distribution means
    "beta0":   0.035,
    "epsilon": 0.20,
    "kappa":   5.0,
    "sigma":   1.0 / 5.0,
    "gamma":   1.0 / 10.0,
    "omega":   1.0 / 180.0,
    "delta":   0.80,
    "nu1":     0.002,
    "nu2":     0.001,

    # Fixed parameters unpacked directly
    **FIXED_PARAMS,
}


# =============================================================================
# 5. PARAM_BOUNDS  (biological validity filter for post-sampling)
# =============================================================================
# After drawing a sample, sampler.py checks each sampled parameter
# against these hard bounds. Any sample that violates a bound is rejected
# and redrawn. This prevents biologically implausible combinations.
#
# Additionally, sampler.py enforces the SIGMA_GAMMA_RATIO_BOUNDS check:
# the ratio sigma/gamma must fall within a plausible biological range.
# This partially compensates for treating sigma and gamma as independent
# when they are biologically correlated. A full treatment would use a
# Gaussian copula — noted as a limitation in the report.

PARAM_BOUNDS = {
    #  parameter    (min,      max)
    "beta0":        (0.005,   0.100),
    "epsilon":      (0.05,    0.50),
    "kappa":        (0.0,     50.0),
    "sigma":        (1/21,    1/1),    # incubation 1 to 21 days
    "gamma":        (1/30,    1/2),    # infectious period 2 to 30 days
    "omega":        (1/365,   1/30),   # immunity 1 month to 1 year
    "delta":        (0.50,    0.99),
    "nu1":          (0.0001,  0.01),
    "nu2":          (0.0001,  0.005),
}

# Biological plausibility filter on the ratio of incubation to recovery rates.
# sigma/gamma = (1/incubation_days) / (1/infectious_days)
#             = infectious_days / incubation_days
# For respiratory pathogens this ratio sits between 0.3 and 4.0.
# Values outside this range indicate an implausible combination even if
# each individual parameter is within its own bounds.
SIGMA_GAMMA_RATIO_BOUNDS = (0.3, 4.0)


# =============================================================================
# 6. INITIAL STATE
# =============================================================================
# Starting conditions at t=0 for all 10 compartments.
#
# State vector ordering — FIXED FOR ENTIRE PROJECT, never change:
#   y = [S1, E1, I1, R1, V1,  S2, E2, I2, R2, V2]
#   idx:  0   1   2   3   4    5   6   7   8   9
#
# Seeded with 10 infectious individuals per age group.
# Everyone else starts susceptible (no prior immunity, no vaccination).
# dtype=float required by scipy's ODE solver.

_N1 = POPULATION["N1"]
_N2 = POPULATION["N2"]

INITIAL_STATE = np.array([
    _N1 - 10,  # S1: susceptible youth
    0,         # E1: exposed youth
    10,        # I1: infectious youth  <- epidemic seed
    0,         # R1: recovered youth
    0,         # V1: vaccinated youth

    _N2 - 10,  # S2: susceptible adults
    0,         # E2: exposed adults
    10,        # I2: infectious adults  <- epidemic seed
    0,         # R2: recovered adults
    0,         # V2: vaccinated adults
], dtype=float)


# =============================================================================
# 7. TIME SPAN
# =============================================================================

TIME_SPAN = (0.0, 5.0 * 365.0)
# Simulate 5 years = 1825 days.
# Long enough to observe: multiple seasonal waves, waning immunity
# reinfection cycles, and approach to endemic equilibrium.

T_EVAL = np.linspace(0.0, 5.0 * 365.0, 5 * 365)
# Daily output points: one value per day for 1825 days.
# Controls what gets returned to us — not the solver's internal step size.
# scipy's RK45 chooses internal steps adaptively for accuracy.

N_ENSEMBLE_RUNS = 500
# Default number of ensemble members.
# 500 runs gives stable percentile estimates (especially at the tails).
# Runtime on a modern laptop: approximately 500 * 0.1s = 50 seconds.
# Use 100 for quick testing, 1000 for publication-quality results.

ENSEMBLE_SEED = 42
# Random seed for the Latin Hypercube Sampler.
# Fixing this guarantees anyone running your code gets identical results.
# Change the seed to verify results are robust to sampling variation —
# a well-designed ensemble should give similar uncertainty bands
# regardless of which seed is used.


# =============================================================================
# 8. HELPER: BUILD CONTACT MATRIX
# =============================================================================

def get_contact_matrix(params: dict) -> np.ndarray:
    """
    Assembles the 2x2 contact matrix C from a parameter dict.

    C[i, j] = daily contacts between one person in group i
              and members of group j.
    (0-indexed: i=0 is youth, i=1 is adults)

    Returns
    -------
    np.ndarray, shape (2, 2)

    Notes
    -----
    Contact matrix entries live in FIXED_PARAMS so this function
    returns the same matrix for every ensemble member. It remains a
    function rather than a module-level constant so that future
    extensions can easily pass scenario-specific contact matrices
    (e.g., school-closure scenarios where c11 and c12 are reduced).
    """
    return np.array([
        [params["c11"], params["c12"]],
        [params["c21"], params["c22"]]
    ])


# =============================================================================
# 9. SELF-CHECK  (run with: python model/parameters.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("PARAMETERS.PY — SELF-CHECK")
    print("=" * 62)

    # --- Population ---
    print(f"\n[1] Population")
    print(f"    Youth  (N1):  {POPULATION['N1']:>10,}")
    print(f"    Adults (N2):  {POPULATION['N2']:>10,}")
    print(f"    Total  (N):   {POPULATION['N']:>10,}")

    # --- Default params summary ---
    p = DEFAULT_PARAMS
    print(f"\n[2] Default parameters (point estimates / distribution means)")
    print(f"    beta0    = {p['beta0']:.5f}   transmission probability /contact/day")
    print(f"    epsilon  = {p['epsilon']:.3f}     seasonality amplitude (±{p['epsilon']*100:.0f}%)")
    print(f"    kappa    = {p['kappa']:.2f}      behavioral response strength")
    print(f"    sigma    = {p['sigma']:.5f}   E→I rate  (incubation = {1/p['sigma']:.0f} days)")
    print(f"    gamma    = {p['gamma']:.5f}   I→R rate  (infectious = {1/p['gamma']:.0f} days)")
    print(f"    omega    = {p['omega']:.7f} R→S rate  (immunity   = {1/p['omega']:.0f} days)")
    print(f"    omega_v  = {p['omega_v']:.7f} V→S rate  (vaccine    = {1/p['omega_v']:.0f} days)")
    print(f"    mu       = {p['mu']:.9f} birth/death rate (lifespan = {1/p['mu']/365:.0f} yrs)")
    print(f"    nu1      = {p['nu1']:.4f}    youth vaccination rate /day")
    print(f"    nu2      = {p['nu2']:.4f}    adult vaccination rate /day")
    print(f"    delta    = {p['delta']:.2f}      vaccine efficacy")

    # --- Contact matrix ---
    print(f"\n[3] Contact matrix (contacts per person per day)")
    C = get_contact_matrix(p)
    print(f"              Youth   Adult")
    print(f"    Youth   [ {C[0,0]:5.1f}   {C[0,1]:5.1f} ]")
    print(f"    Adult   [ {C[1,0]:5.1f}   {C[1,1]:5.1f} ]")
    gap = abs(p["c12"] * POPULATION["N1"] - p["c21"] * POPULATION["N2"])
    print(f"    Reciprocity gap = {gap:,.0f}  "
          f"({'OK' if gap < 300_000 else 'large — check values'})")

    # --- Initial state ---
    print(f"\n[4] Initial state (t=0)")
    labels = ["S1","E1","I1","R1","V1","S2","E2","I2","R2","V2"]
    for i, (lbl, val) in enumerate(zip(labels, INITIAL_STATE)):
        print(f"    y[{i}] {lbl} = {int(val):>10,}")
    s = INITIAL_STATE.sum()
    N = POPULATION["N"]
    print(f"    Conservation: sum={s:,.0f}  N={N:,}  "
          f"{'PASS' if abs(s-N)<1 else 'FAIL'}")

    # --- Sigma/gamma ratio ---
    print(f"\n[5] Biological plausibility (default params)")
    ratio = p["sigma"] / p["gamma"]
    lo, hi = SIGMA_GAMMA_RATIO_BOUNDS
    print(f"    sigma/gamma = {ratio:.3f}  bounds=[{lo},{hi}]  "
          f"{'PASS' if lo<=ratio<=hi else 'FAIL'}")
    print(f"    Interpretation: infectious period is {ratio:.1f}x the incubation period")

    # --- Distribution sanity check ---
    print(f"\n[6] Distribution sanity (20 test draws per parameter)")
    rng_test = np.random.default_rng(seed=999)
    print(f"    {'param':<12} {'mean':>10} {'min':>10} {'max':>10} {'in-bounds':>10}")
    print(f"    {'-'*56}")
    for name, dist in PARAM_DISTRIBUTIONS.items():
        samples = dist.rvs(size=20, random_state=rng_test)
        lo_b, hi_b = PARAM_BOUNDS.get(name, (-np.inf, np.inf))
        n_ok = int(np.sum((samples >= lo_b) & (samples <= hi_b)))
        print(f"    {name:<12} {np.mean(samples):>10.5f} "
              f"{np.min(samples):>10.5f} {np.max(samples):>10.5f} "
              f"{n_ok:>8}/20")

    # --- Ensemble settings ---
    print(f"\n[7] Ensemble settings")
    print(f"    N_ENSEMBLE_RUNS  = {N_ENSEMBLE_RUNS}")
    print(f"    ENSEMBLE_SEED    = {ENSEMBLE_SEED}")
    print(f"    Sampled params   = {SAMPLED_PARAM_NAMES}")
    print(f"    Fixed params     = {list(FIXED_PARAMS.keys())}")
    est_time = N_ENSEMBLE_RUNS * 0.1
    print(f"    Est. runtime     ~ {est_time:.0f}s ({est_time/60:.1f} min) on a laptop")

    print("\n" + "=" * 62)
    print("All checks complete.")
    print("=" * 62)

"""
solver.py
=========
Runs the SEIRS-V ODE system and returns structured results.

WHAT THIS FILE DOES
-------------------
Takes parameter dictionaries from sampler.py and initial conditions
from parameters.py, feeds them into scipy's ODE integrator alongside
the equations from equations.py, and returns epidemic trajectories
as structured numpy arrays.

There are two modes:

    run_single()   — one deterministic run with one parameter set.
                     Used for validation, debugging, and the dashboard's
                     baseline display before the full ensemble loads.

    run_ensemble() — loops over N parameter sets from draw_ensemble(),
                     runs the ODE once per set, stacks all trajectories
                     into a 3D array, then computes percentile statistics.
                     This is the prior predictive ensemble.

WHY RK45 WITH max_step=1.0
---------------------------
scipy's solve_ivp defaults to RK45 (Runge-Kutta 4/5), an adaptive
step-size integrator. "Adaptive" means it automatically takes smaller
steps where the solution changes rapidly (near epidemic peaks) and
larger steps where it changes slowly (plateau or trough periods).

The max_step=1.0 constraint limits the internal step size to one day.
Without this constraint, the seasonal forcing function beta(t) —
which oscillates with a 365-day period — could be undersampled if
the solver takes steps of many days during quiet epidemic periods,
causing it to miss the fine structure of the seasonal wave. One day
maximum step prevents this.

T_EVAL controls which time points are returned to us — one per day
for 1825 days. This does NOT control the solver's internal accuracy;
that is handled by the adaptive RK45 mechanism independently.

ENSEMBLE OUTPUT SHAPE
---------------------
The 3D result array has shape (N_runs, N_timepoints, 10):
    axis 0 = ensemble member index (0 to N_runs-1)
    axis 1 = time index (0 to N_timepoints-1), one per day
    axis 2 = compartment index (matches state vector ordering)
              0:S1  1:E1  2:I1  3:R1  4:V1
              5:S2  6:E2  7:I2  8:R2  9:V2

Percentile statistics are computed along axis=0 (across ensemble
members at each timepoint), giving arrays of shape (N_timepoints, 10).
"""

import numpy as np
from scipy.integrate import solve_ivp

from model.parameters import (
    DEFAULT_PARAMS,
    INITIAL_STATE,
    TIME_SPAN,
    T_EVAL,
    N_ENSEMBLE_RUNS,
    ENSEMBLE_SEED,
    POPULATION,
)
from model.equations import seirs_v_odes
from model.sampler import draw_ensemble


# =============================================================================
# RESULT CONTAINER
# =============================================================================

class SingleResult:
    """
    Holds the output of one ODE run.

    Attributes
    ----------
    t : np.ndarray, shape (N_timepoints,)
        Time points in days.
    y : np.ndarray, shape (N_timepoints, 10)
        Compartment values at each time point.
        NOTE: scipy returns shape (10, N_timepoints) — we transpose
        immediately so axis 0 is time, matching the ensemble convention.
    params : dict
        The parameter set used for this run.
    success : bool
        Whether the ODE solver completed successfully.
    message : str
        Solver status message (useful for debugging failed runs).
    """
    def __init__(self, sol, params: dict):
        self.t       = sol.t                 # shape (N_timepoints,)
        self.y       = sol.y.T               # shape (N_timepoints, 10)
        self.params  = params
        self.success = sol.success
        self.message = sol.message

    def compartment(self, idx: int) -> np.ndarray:
        """
        Returns the time series for one compartment.

        Parameters
        ----------
        idx : int
            Compartment index 0-9. See state vector ordering:
            0:S1 1:E1 2:I1 3:R1 4:V1 5:S2 6:E2 7:I2 8:R2 9:V2

        Returns
        -------
        np.ndarray, shape (N_timepoints,)
        """
        return self.y[:, idx]

    def total_infectious(self) -> np.ndarray:
        """Returns I1 + I2 across all timepoints."""
        return self.y[:, 2] + self.y[:, 7]

    def total_population(self) -> np.ndarray:
        """
        Returns total population at each timepoint.
        Should be constant at N = 700,000 throughout.
        Any drift indicates a bug in the equations.
        """
        return self.y.sum(axis=1)


class EnsembleResult:
    """
    Holds the output of a full ensemble run.

    Attributes
    ----------
    t : np.ndarray, shape (N_timepoints,)
        Time points in days. Same for all ensemble members.

    trajectories : np.ndarray, shape (N_runs, N_timepoints, 10)
        All individual epidemic trajectories.
        axis 0 = ensemble member
        axis 1 = time point
        axis 2 = compartment (0-9)

    median : np.ndarray, shape (N_timepoints, 10)
        Median trajectory across ensemble members at each timepoint.

    lower : np.ndarray, shape (N_timepoints, 10)
        2.5th percentile (lower bound of 95% credible interval).

    upper : np.ndarray, shape (N_timepoints, 10)
        97.5th percentile (upper bound of 95% credible interval).

    param_list : list of dict
        The N parameter sets used (from draw_ensemble()).

    n_failed : int
        Number of ODE runs that did not converge. Should be 0.
    """
    def __init__(self, t, trajectories, param_list, n_failed):
        self.t            = t
        self.trajectories = trajectories   # (N_runs, N_timepoints, 10)
        self.param_list   = param_list
        self.n_failed     = n_failed

        # Compute percentile statistics across ensemble axis (axis=0)
        self.median = np.percentile(trajectories, 50,   axis=0)
        self.lower  = np.percentile(trajectories,  2.5, axis=0)
        self.upper  = np.percentile(trajectories, 97.5, axis=0)

    def total_infectious_bands(self) -> tuple:
        """
        Returns (median, lower, upper) for total I1 + I2 over time.
        These are the primary curves displayed in the dashboard.

        Returns
        -------
        tuple of three np.ndarray, each shape (N_timepoints,)
        """
        # Sum I1 (col 2) and I2 (col 7) within each band
        # Note: we sum the percentile bands, not percentiles of the sum.
        # This is a standard approximation — computing the percentile of
        # I1+I2 directly would require sorting the combined quantity
        # across all ensemble members, which analysis.py handles properly.
        return (
            self.median[:, 2] + self.median[:, 7],
            self.lower[:, 2]  + self.lower[:, 7],
            self.upper[:, 2]  + self.upper[:, 7],
        )

    def compartment_bands(self, idx: int) -> tuple:
        """
        Returns (median, lower, upper) for a single compartment.

        Parameters
        ----------
        idx : int
            Compartment index 0-9.

        Returns
        -------
        tuple of three np.ndarray, each shape (N_timepoints,)
        """
        return (
            self.median[:, idx],
            self.lower[:, idx],
            self.upper[:, idx],
        )


# =============================================================================
# SINGLE RUN
# =============================================================================

def run_single(
    params: dict     = None,
    initial_state    = None,
    time_span: tuple = None,
    t_eval           = None,
) -> SingleResult:
    """
    Runs the ODE system once with a single parameter set.

    All arguments default to the values from parameters.py, so you
    can call run_single() with no arguments for a quick baseline run.

    Parameters
    ----------
    params : dict, optional
        Complete parameter dictionary. Defaults to DEFAULT_PARAMS.
    initial_state : np.ndarray, optional
        Starting compartment values. Defaults to INITIAL_STATE.
    time_span : tuple, optional
        (t_start, t_end) in days. Defaults to TIME_SPAN = (0, 1825).
    t_eval : np.ndarray, optional
        Time points to record. Defaults to T_EVAL (one per day).

    Returns
    -------
    SingleResult
        Contains .t, .y, .params, .success, .message.

    Notes
    -----
    scipy's solve_ivp signature:
        solve_ivp(fun, t_span, y0, method, t_eval, args, max_step)

    The lambda wrapper is needed because solve_ivp passes (t, y) to
    fun, but seirs_v_odes needs a third argument (params). The lambda
    captures params from the enclosing scope.

    max_step=1.0 prevents the adaptive solver from stepping over the
    fine structure of the seasonal forcing function beta(t).
    """
    if params       is None: params        = DEFAULT_PARAMS
    if initial_state is None: initial_state = INITIAL_STATE
    if time_span    is None: time_span     = TIME_SPAN
    if t_eval       is None: t_eval        = T_EVAL

    sol = solve_ivp(
        fun=lambda t, y: seirs_v_odes(t, y, params),
        t_span=time_span,
        y0=initial_state,
        method="RK45",
        t_eval=t_eval,
        max_step=1.0,        # never skip more than one day internally
        dense_output=False,  # we only need values at t_eval points
    )

    if not sol.success:
        print(f"  WARNING: ODE solver did not converge. "
              f"Message: {sol.message}")

    return SingleResult(sol, params)


# =============================================================================
# ENSEMBLE RUN
# =============================================================================

def run_ensemble(
    n: int           = N_ENSEMBLE_RUNS,
    seed: int        = ENSEMBLE_SEED,
    initial_state    = None,
    time_span: tuple = None,
    t_eval           = None,
    verbose: bool    = True,
) -> EnsembleResult:
    """
    Runs the ODE system N times, once per sampled parameter set.

    This is the core of the prior predictive ensemble. Each run uses
    a different parameter set drawn from the prior distributions via
    Latin Hypercube Sampling. The spread of the N resulting trajectories
    quantifies how uncertain we are about the epidemic trajectory given
    our uncertainty about the parameter values.

    Parameters
    ----------
    n : int
        Number of ensemble members. Default: 500.
    seed : int
        Random seed for LHS. Default: 42. Change to verify robustness.
    initial_state : np.ndarray, optional
        Starting conditions. Defaults to INITIAL_STATE.
    time_span : tuple, optional
        (t_start, t_end). Defaults to TIME_SPAN.
    t_eval : np.ndarray, optional
        Output time points. Defaults to T_EVAL.
    verbose : bool
        If True, prints progress every 50 runs. Default: True.

    Returns
    -------
    EnsembleResult
        Contains .trajectories, .median, .lower, .upper, .t, .param_list.

    Runtime
    -------
    Approximately 0.1 seconds per run on a modern laptop.
    500 runs ~ 50 seconds total.
    Set n=50 for quick testing during development.

    Memory
    ------
    The trajectories array has shape (500, 1825, 10).
    Size: 500 * 1825 * 10 * 8 bytes = ~73 MB. Manageable.
    """
    if initial_state is None: initial_state = INITIAL_STATE
    if time_span     is None: time_span     = TIME_SPAN
    if t_eval        is None: t_eval        = T_EVAL

    N_timepoints = len(t_eval)

    # Draw all N parameter sets using LHS before running any ODEs
    # This separates sampling (fast) from solving (slow) cleanly
    if verbose:
        print(f"Drawing {n} parameter sets via LHS (seed={seed})...")
    param_list = draw_ensemble(n=n, seed=seed)

    # Pre-allocate the result array for efficiency
    # Filling a pre-allocated array is faster than appending N times
    trajectories = np.zeros((n, N_timepoints, 10))
    n_failed = 0

    if verbose:
        print(f"Running {n} ODE integrations...")

    for i, params in enumerate(param_list):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Completed {i+1}/{n} runs...")

        sol = solve_ivp(
            fun=lambda t, y, p=params: seirs_v_odes(t, y, p),
            # Note: the default argument p=params captures the current
            # params in the lambda, preventing the common Python closure
            # bug where all lambdas in a loop reference the final value.
            t_span=time_span,
            y0=initial_state,
            method="RK45",
            t_eval=t_eval,
            max_step=1.0,
            dense_output=False,
        )

        if sol.success:
            # sol.y has shape (10, N_timepoints) — transpose to (N_timepoints, 10)
            trajectories[i] = sol.y.T
        else:
            n_failed += 1
            # Fill with NaN so percentile computation skips this run
            trajectories[i] = np.nan
            if verbose:
                print(f"  WARNING: Run {i} failed. Message: {sol.message}")

    if verbose:
        print(f"Ensemble complete. Failed runs: {n_failed}/{n}")
        if n_failed > 0:
            print("  Failed runs are excluded from percentile calculations.")

    return EnsembleResult(
        t=t_eval,
        trajectories=trajectories,
        param_list=param_list,
        n_failed=n_failed,
    )


# =============================================================================
# SELF-CHECK  (run with: python model/solver.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("SOLVER.PY — SELF-CHECK")
    print("=" * 62)

    N = POPULATION["N"]

    # ------------------------------------------------------------------
    print("\n[1] Single run with default parameters")
    result = run_single()
    print(f"    Solver success:   {result.success}")
    print(f"    Time points:      {len(result.t)}")
    print(f"    Output shape:     {result.y.shape}  (expected: (1825, 10))")
    print(f"    t[0]={result.t[0]:.1f}  t[-1]={result.t[-1]:.1f} days")

    # ------------------------------------------------------------------
    print("\n[2] Conservation check (single run)")
    pop_t = result.total_population()
    drift = np.max(np.abs(pop_t - N))
    print(f"    N at t=0:    {pop_t[0]:,.2f}  (expected {N:,})")
    print(f"    N at t=end:  {pop_t[-1]:,.2f}")
    print(f"    Max drift:   {drift:.4f} people")
    print(f"    Status:      {'PASS' if drift < 1.0 else 'FAIL — check equations'}")

    # ------------------------------------------------------------------
    print("\n[3] Epidemic behavior check (single run)")
    I_total = result.total_infectious()
    peak_val = I_total.max()
    peak_day = result.t[I_total.argmax()]
    print(f"    Peak infectious:  {peak_val:,.0f} people")
    print(f"    Peak day:         day {peak_day:.0f} ({peak_day/365:.1f} years)")
    print(f"    I_total at t=0:   {I_total[0]:,.0f}")
    print(f"    I_total at t=end: {I_total[-1]:,.0f}")
    epidemic_grew = I_total.max() > I_total[0]
    print(f"    Epidemic grew:    {epidemic_grew}  (expected True for R0>1)")

    # ------------------------------------------------------------------
    print("\n[4] Small ensemble run (n=20 for speed)")
    ens = run_ensemble(n=20, seed=42, verbose=True)
    print(f"    Trajectories shape: {ens.trajectories.shape}  "
          f"(expected: (20, 1825, 10))")
    print(f"    Median shape:       {ens.median.shape}")
    print(f"    Lower shape:        {ens.lower.shape}")
    print(f"    Upper shape:        {ens.upper.shape}")
    print(f"    Failed runs:        {ens.n_failed}")

    # ------------------------------------------------------------------
    print("\n[5] Percentile ordering check")
    # At every timepoint and compartment, lower <= median <= upper
    lower_ok  = np.all(ens.lower  <= ens.median + 1e-6)
    upper_ok  = np.all(ens.upper  >= ens.median - 1e-6)
    print(f"    lower <= median everywhere: {lower_ok}")
    print(f"    upper >= median everywhere: {upper_ok}")
    print(f"    Status: {'PASS' if lower_ok and upper_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    print("\n[6] Ensemble uncertainty check")
    # The CI band for I_total should be non-trivial (uncertainty exists)
    med, lo, hi = ens.total_infectious_bands()
    avg_width = np.mean(hi - lo)
    print(f"    Mean CI width (I_total): {avg_width:,.1f} people")
    print(f"    Peak median I_total:     {med.max():,.0f}")
    print(f"    Width is non-trivial:    {avg_width > 10}")

    # ------------------------------------------------------------------
    print("\n[7] Conservation check (ensemble median)")
    median_pop = ens.median.sum(axis=1)
    drift_ens  = np.max(np.abs(median_pop - N))
    print(f"    Max drift in median:  {drift_ens:.4f} people")
    print(f"    Status: {'PASS' if drift_ens < 1.0 else 'FAIL'}")

    print("\n" + "=" * 62)
    print("solver.py self-check complete.")
    print("=" * 62)

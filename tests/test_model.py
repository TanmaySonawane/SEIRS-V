"""
test_model.py
=============
Automated validation suite for the SEIRS-V epidemic model.

HOW TO RUN
----------
From the project root directory:

    pytest tests/
    pytest tests/ -v                  # verbose: shows each test name
    pytest tests/ -v --tb=short       # short traceback on failures
    pytest tests/test_model.py::TestConservation -v   # one class only

WHAT IS BEING TESTED
---------------------
These tests implement the four validation checks described in the
project report, plus additional sanity checks for the ensemble
and the probabilistic components.

The four core validation checks are:

    CHECK 1 — Population conservation
        The sum of all compartments must equal N = 700,000 at every
        single timestep. Any drift means there is a sign error in the
        ODEs — a flow that appears in one equation but not its counterpart.

    CHECK 2 — Degenerate case recovery
        When all novel features are turned off (epsilon=0, kappa=0,
        nu=0, omega=0, age groups made identical), the model must
        reproduce the standard scalar SEIR result. If a simplified
        model matches a known solution, it proves the complex model
        contains the simple one as a correct special case.

    CHECK 3 — Threshold behavior
        When R0 < 1, the epidemic must die out (I → 0).
        When R0 > 1, the epidemic must establish and grow initially.
        This proves the R0 formula actually governs model behavior.

    CHECK 4 — Directional sensitivity
        Every parameter must move the output in the biologically
        expected direction: higher vaccination → lower peak, higher
        kappa → lower peak, higher epsilon → higher wave amplitude,
        higher omega → more recurrent waves.

WHY THESE TESTS MATTER FOR A NOVEL MODEL
-----------------------------------------
Unlike reproducing a published model (where you can compare plots),
this model is a novel combination. No paper shows exactly these
results. The validation strategy is therefore:

    "Prove the model obeys mathematical laws that ALL models of
     this class must satisfy, regardless of parameter values."

A model that passes all four checks is not guaranteed to be correct
in every respect, but it is guaranteed to be internally consistent
and to behave sensibly at its boundaries. That is the strongest
possible validation for a novel model without empirical data.
"""

import numpy as np
import pytest

from model.parameters import (
    DEFAULT_PARAMS,
    FIXED_PARAMS,
    POPULATION,
    INITIAL_STATE,
    TIME_SPAN,
    T_EVAL,
    PARAM_BOUNDS,
    SIGMA_GAMMA_RATIO_BOUNDS,
    N_ENSEMBLE_RUNS,
    ENSEMBLE_SEED,
)
from model.equations  import (
    seirs_v_odes,
    seasonal_beta,
    behavioral_factor,
    force_of_infection,
)
from model.sampler    import draw_single_sample, draw_ensemble, ensemble_to_array
from model.solver     import run_single, run_ensemble
from model.analysis   import (
    compute_r0,
    compute_dfe,
    compute_ngm,
    compute_peak_stats,
    compute_final_size,
    compute_r0_ensemble,
)


# =============================================================================
# SHARED FIXTURES
# =============================================================================
# pytest fixtures are functions that provide test data.
# By marking them with @pytest.fixture, pytest injects them into any
# test function that lists them as parameters — no need to repeat setup.

@pytest.fixture(scope="module")
def default_single_run():
    """
    Runs the model once with DEFAULT_PARAMS.
    scope="module" means this runs once per test file, not once per test.
    All tests in this file that need a single run share this result.
    """
    return run_single(params=DEFAULT_PARAMS)


@pytest.fixture(scope="module")
def small_ensemble():
    """
    Runs a small ensemble (n=50) for ensemble-level tests.
    50 runs is enough for structural checks without taking too long.
    """
    return run_ensemble(n=50, seed=ENSEMBLE_SEED, verbose=False)


@pytest.fixture(scope="module")
def small_param_list():
    """Returns 50 sampled parameter sets for analysis tests."""
    return draw_ensemble(n=50, seed=ENSEMBLE_SEED)


# =============================================================================
# CHECK 1 — POPULATION CONSERVATION
# =============================================================================

class TestConservation:
    """
    The total population N must remain constant at all times.
    This is the most fundamental test — it verifies that the ODEs
    are correctly balanced (every outflow has a matching inflow).
    """

    def test_initial_state_sums_to_N(self):
        """
        The initial conditions must sum to N = 700,000.
        If this fails, INITIAL_STATE in parameters.py is wrong.
        """
        N = POPULATION["N"]
        total = INITIAL_STATE.sum()
        assert abs(total - N) < 1.0, (
            f"Initial state sums to {total:.2f}, expected {N}. "
            f"Check INITIAL_STATE in parameters.py."
        )

    def test_conservation_throughout_simulation(self, default_single_run):
        """
        Population must not drift at any point during the 5-year run.
        Tolerance of 1.0 person accounts for floating-point rounding
        in the ODE solver — any drift beyond 1 person indicates a
        structural error in the equations.
        """
        N = POPULATION["N"]
        result = default_single_run
        pop_at_each_step = result.y.sum(axis=1)  # sum across 10 compartments

        max_drift = np.max(np.abs(pop_at_each_step - N))
        assert max_drift < 1.0, (
            f"Population drifted by {max_drift:.4f} people during simulation. "
            f"There is a sign error or missing term in equations.py."
        )

    def test_conservation_youth_group(self, default_single_run):
        """Youth compartments (indices 0-4) must sum to N1 = 200,000."""
        N1 = POPULATION["N1"]
        result = default_single_run
        youth_total = result.y[:, 0:5].sum(axis=1)
        max_drift = np.max(np.abs(youth_total - N1))
        assert max_drift < 1.0, (
            f"Youth population drifted by {max_drift:.4f}. "
            f"Check youth ODE terms in equations.py."
        )

    def test_conservation_adult_group(self, default_single_run):
        """Adult compartments (indices 5-9) must sum to N2 = 500,000."""
        N2 = POPULATION["N2"]
        result = default_single_run
        adult_total = result.y[:, 5:10].sum(axis=1)
        max_drift = np.max(np.abs(adult_total - N2))
        assert max_drift < 1.0, (
            f"Adult population drifted by {max_drift:.4f}. "
            f"Check adult ODE terms in equations.py."
        )

    def test_ode_derivatives_sum_to_zero(self):
        """
        At any state, the sum of all 10 derivatives must equal zero.
        This is the algebraic proof of conservation:
            d/dt[sum of all compartments] = sum of all derivatives = 0
        because births (Lambda_i = mu*N_i) exactly cancel deaths (mu*N_i).

        We test this at t=0 with the default initial state.
        A nonzero sum means there is an unbalanced term in the equations.
        """
        dydt = seirs_v_odes(0.0, INITIAL_STATE, DEFAULT_PARAMS)
        total_rate = dydt.sum()
        assert abs(total_rate) < 1e-6, (
            f"Sum of all derivatives = {total_rate:.2e}, expected ~0. "
            f"Births and deaths are not balanced in equations.py."
        )

    def test_conservation_ensemble(self, small_ensemble):
        """Conservation must hold for every individual ensemble member."""
        N = POPULATION["N"]
        # Check each run independently — per-compartment medians do not
        # sum to N because they come from different ensemble members.
        for i, traj in enumerate(small_ensemble.trajectories):
            pop_at_each_step = traj.sum(axis=1)
            max_drift = np.max(np.abs(pop_at_each_step - N))
            assert max_drift < 2.0, (
                f"Ensemble member {i} drifted by {max_drift:.4f}."
            )


# =============================================================================
# CHECK 2 — DEGENERATE CASE RECOVERY
# =============================================================================

class TestDegenerateCase:
    """
    When all novel features are disabled, the model must reduce to
    the standard SEIR result.

    This is the most powerful structural test: it proves the complex
    model correctly contains the simple model as a special case.

    Standard SEIR has:
        - No seasonality (epsilon=0)
        - No behavioral adaptation (kappa=0)
        - No vaccination (nu=0)
        - No waning immunity (omega=0) — permanent immunity after recovery
        - No vital dynamics (mu=0) — no births or deaths
        - No age structure — both groups identical

    Under these conditions, the 10-ODE system must behave like
    the scalar SEIR with R0 = beta0 / gamma.
    """

    @pytest.fixture
    def seir_params(self):
        """
        Parameter set that reduces our model to standard SEIR.
        All novel features set to zero or neutralized.
        """
        p = DEFAULT_PARAMS.copy()
        p["epsilon"] = 0.0      # no seasonality
        p["kappa"]   = 0.0      # no behavioral adaptation
        p["nu1"]     = 0.0      # no youth vaccination
        p["nu2"]     = 0.0      # no adult vaccination
        p["omega"]   = 0.0      # permanent immunity (no waning)
        p["mu"]      = 0.0      # no births or deaths
        p["beta0"]   = 0.5      # R0 = beta0/gamma = 0.5/0.1 = 5 > 1
        # Zero cross-group contacts so each group is an independent scalar
        # SEIR. With c_12=c_21=0, K is diagonal and R0 = beta0*c_ii/gamma
        # exactly, matching the single-group scalar formula.
        # Using c_ij=1 for all i,j with N1≠N2 gives R0=2*beta0/gamma
        # because the two unequal groups create asymmetric NGM entries.
        p["c11"] = 1.0
        p["c12"] = 0.0
        p["c21"] = 0.0
        p["c22"] = 1.0
        return p

    def test_r0_matches_scalar_seir(self, seir_params):
        """
        R0 from NGM must match the scalar SEIR formula R0 = beta0/gamma
        when all complexity is removed and mu=0.

        Scalar SEIR R0 = beta0 * (1/gamma) because:
            - One infectious person generates beta0 * c contacts per day
            - Duration of infectiousness = 1/gamma days
            - All contacts are with susceptibles (DFE)
        """
        R0_model  = compute_r0(seir_params)
        # With uniform contacts (all c_ij=1) and no vaccination (V*=0),
        # all contacts are susceptible, so:
        # R0 = beta0 * 1 * sigma/((sigma+0)*(gamma+0)) * N/N = beta0/gamma
        R0_scalar = seir_params["beta0"] / seir_params["gamma"]

        assert abs(R0_model - R0_scalar) < 0.01, (
            f"NGM R0 = {R0_model:.4f} but scalar SEIR R0 = {R0_scalar:.4f}. "
            f"Difference of {abs(R0_model-R0_scalar):.4f} exceeds tolerance. "
            f"Check the NGM computation in analysis.py."
        )

    def test_epidemic_grows_when_r0_above_one(self, seir_params):
        """
        Under SEIR conditions with R0 > 1, infectious count must
        initially increase. The epidemic must grow before it peaks.
        """
        result = run_single(params=seir_params)
        I_total = result.y[:, 2] + result.y[:, 7]
        # Check that I_total increases at some point before day 365
        early_max = I_total[:365].max()
        assert early_max > INITIAL_STATE[2] + INITIAL_STATE[7], (
            f"Epidemic did not grow under SEIR conditions with R0 > 1. "
            f"Initial I = {INITIAL_STATE[2]+INITIAL_STATE[7]:.0f}, "
            f"early max I = {early_max:.0f}."
        )

    def test_no_seasonality_when_epsilon_zero(self):
        """
        When epsilon=0, beta(t) must be constant regardless of t.
        The seasonal forcing function must return beta0 exactly.
        """
        p = DEFAULT_PARAMS.copy()
        p["epsilon"] = 0.0

        times = [0, 91, 182, 273, 365, 730]
        betas = [seasonal_beta(t, p) for t in times]

        for t, b in zip(times, betas):
            assert abs(b - p["beta0"]) < 1e-12, (
                f"At t={t}, beta={b:.8f} but beta0={p['beta0']:.8f}. "
                f"Seasonality not suppressed when epsilon=0."
            )

    def test_no_behavioral_dampening_when_kappa_zero(self):
        """
        When kappa=0, Phi must equal 1.0 at all prevalence levels.
        No behavioral adaptation should occur.
        """
        p = DEFAULT_PARAMS.copy()
        p["kappa"] = 0.0
        N = POPULATION["N"]

        for prevalence in [0.0, 0.01, 0.05, 0.10, 0.30]:
            I_total = prevalence * N
            phi = behavioral_factor(I_total, N, p)
            assert abs(phi - 1.0) < 1e-12, (
                f"kappa=0 but Phi={phi:.8f} at prevalence={prevalence*100:.0f}%. "
                f"Behavioral dampening not suppressed."
            )


# =============================================================================
# CHECK 3 — THRESHOLD BEHAVIOR
# =============================================================================

class TestThresholdBehavior:
    """
    R0 must govern whether the epidemic establishes or dies out.
    R0 > 1: epidemic grows.
    R0 < 1: epidemic dies out.

    This proves the R0 formula actually controls model behavior —
    not just that the formula computes a number, but that the number
    means what it is supposed to mean.
    """

    @pytest.fixture
    def high_vaccination_params(self):
        """
        High vaccination rate and efficacy designed to push R0 below 1.
        With enough vaccinated people, there are not enough susceptibles
        to sustain transmission.
        """
        p = DEFAULT_PARAMS.copy()
        p["nu1"]   = 0.01    # 1% of susceptible youth vaccinated per day
        p["nu2"]   = 0.01    # 1% of susceptible adults per day
        p["delta"] = 0.99    # near-perfect vaccine
        p["kappa"] = 0.0     # disable behavioral adaptation for clean test
        p["epsilon"] = 0.0   # disable seasonality for clean test
        return p

    def test_r0_below_one_with_high_vaccination(self, high_vaccination_params):
        """R0 must be below 1 with strong vaccination parameters."""
        R0 = compute_r0(high_vaccination_params)
        assert R0 < 1.0, (
            f"Expected R0 < 1 with high vaccination, got R0 = {R0:.4f}. "
            f"Increase nu1, nu2, or delta."
        )

    def test_epidemic_dies_out_when_r0_below_one(self, high_vaccination_params):
        """
        When R0 < 1, infectious count must be lower at the end of the
        simulation than at the start. The epidemic cannot sustain itself.

        We use a shorter simulation (1 year) to keep the test fast.
        Note: with vital dynamics (mu > 0), new susceptibles are born
        continuously, so the epidemic may not go to exactly zero —
        but it must clearly decline from the initial seed.
        """
        # Run full 5 years: R0<1 at the DFE but the simulation starts with
        # zero vaccinated people, so the epidemic grows initially while
        # vaccination accumulates. By year 5 the DFE is reached and I→0.
        long_t = np.linspace(0, 1825, 1825)
        result  = run_single(
            params=high_vaccination_params,
            t_eval=long_t,
            time_span=(0, 1825),
        )
        I_initial = result.y[0,  2] + result.y[0,  7]
        I_final   = result.y[-1, 2] + result.y[-1, 7]

        assert I_final < I_initial, (
            f"Epidemic grew despite R0 < 1. "
            f"Initial I = {I_initial:.0f}, Final I = {I_final:.0f}. "
            f"Check that R0 formula and ODE dynamics are consistent."
        )

    def test_epidemic_grows_when_r0_above_one(self, default_single_run):
        """
        With DEFAULT_PARAMS (R0 > 1), the epidemic must grow from the seed.
        Check that peak I_total exceeds the initial seed of 20 people
        by a substantial margin.
        """
        R0 = compute_r0(DEFAULT_PARAMS)
        assert R0 > 1.0, (
            f"DEFAULT_PARAMS should give R0 > 1, got {R0:.4f}. "
            f"Increase beta0 or adjust parameters."
        )

        result  = default_single_run
        I_total = result.y[:, 2] + result.y[:, 7]
        I_seed  = INITIAL_STATE[2] + INITIAL_STATE[7]   # = 20 people

        assert I_total.max() > I_seed * 10, (
            f"Epidemic barely grew from seed of {I_seed:.0f}. "
            f"Peak = {I_total.max():.0f}. "
            f"Expected at least {I_seed*10:.0f} with R0 = {R0:.3f}."
        )

    def test_kappa_does_not_affect_r0(self):
        """
        R0 must be identical regardless of kappa value.
        Behavioral adaptation has no effect at the disease-free equilibrium
        because prevalence is zero there, making Phi = 1 exactly.

        This is one of the key analytical results of the project.
        """
        R0_values = []
        for kappa_test in [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]:
            p = DEFAULT_PARAMS.copy()
            p["kappa"] = kappa_test
            R0_values.append(compute_r0(p))

        # All R0 values must be identical to machine precision
        for i, (k, r) in enumerate(zip([0.0,1.0,5.0,10.0,50.0,100.0], R0_values)):
            assert abs(r - R0_values[0]) < 1e-10, (
                f"R0 changed with kappa. "
                f"kappa=0 gives R0={R0_values[0]:.8f}, "
                f"kappa={k} gives R0={r:.8f}. "
                f"kappa must not appear in the NGM calculation."
            )

    def test_r0_formula_matches_eigenvalue(self):
        """
        The closed-form R0 formula and numpy's eigenvalue must agree.
        This cross-validates the NGM computation in analysis.py.
        """
        dfe = compute_dfe(DEFAULT_PARAMS)
        K   = compute_ngm(DEFAULT_PARAMS, dfe)

        # Closed-form for 2x2 dominant eigenvalue
        K11, K12, K21, K22 = K[0,0], K[0,1], K[1,0], K[1,1]
        disc     = ((K11 - K22) / 2.0) ** 2 + K12 * K21
        R0_form  = (K11 + K22) / 2.0 + np.sqrt(max(disc, 0.0))

        # Numpy eigenvalue
        R0_eig = float(np.max(np.linalg.eigvals(K).real))

        assert abs(R0_form - R0_eig) < 1e-8, (
            f"R0 formula ({R0_form:.8f}) disagrees with eigenvalue ({R0_eig:.8f}). "
            f"Difference: {abs(R0_form-R0_eig):.2e}."
        )


# =============================================================================
# CHECK 4 — DIRECTIONAL SENSITIVITY
# =============================================================================

class TestDirectionalSensitivity:
    """
    Each parameter must move the model output in the biologically
    expected direction. If a parameter moves output the wrong way,
    there is a sign error somewhere in the equations or analysis.

    These tests do not check magnitudes — only directions (more/less).
    """

    def _run_comparison(self, param_name: str, low_val, high_val,
                        output_fn, expect_higher_output_for_higher_param: bool):
        """
        Helper: runs the model twice (low and high parameter values),
        computes an output metric for each, and checks the direction.

        Parameters
        ----------
        param_name : str
        low_val, high_val : float
        output_fn : callable
            Takes a SingleResult and returns a scalar metric.
        expect_higher_output_for_higher_param : bool
            True if metric should increase when param increases.
            False if metric should decrease when param increases.
        """
        p_low  = {**DEFAULT_PARAMS, param_name: low_val,
                  "epsilon": 0.0, "kappa": 0.0}  # isolate the parameter
        p_high = {**DEFAULT_PARAMS, param_name: high_val,
                  "epsilon": 0.0, "kappa": 0.0}

        r_low  = run_single(params=p_low)
        r_high = run_single(params=p_high)

        out_low  = output_fn(r_low)
        out_high = output_fn(r_high)

        if expect_higher_output_for_higher_param:
            assert out_high > out_low, (
                f"Higher {param_name} should increase output. "
                f"{param_name}={low_val} → {out_low:.2f}, "
                f"{param_name}={high_val} → {out_high:.2f}."
            )
        else:
            assert out_high < out_low, (
                f"Higher {param_name} should decrease output. "
                f"{param_name}={low_val} → {out_low:.2f}, "
                f"{param_name}={high_val} → {out_high:.2f}."
            )

    def _peak_I_total(self, result):
        """Peak total infectious count across simulation."""
        return (result.y[:, 2] + result.y[:, 7]).max()

    def test_higher_beta0_increases_peak(self):
        """Higher transmission → more infections at peak."""
        self._run_comparison(
            "beta0", low_val=0.02, high_val=0.05,
            output_fn=self._peak_I_total,
            expect_higher_output_for_higher_param=True
        )

    def test_higher_vaccination_decreases_peak(self):
        """
        Higher vaccination rate → fewer susceptibles → lower peak.
        We test nu1 (youth rate) since youth drive more transmission
        due to higher c11 contact rate.
        """
        p_low  = {**DEFAULT_PARAMS, "nu1": 0.0001, "nu2": 0.0001,
                  "epsilon": 0.0, "kappa": 0.0}
        p_high = {**DEFAULT_PARAMS, "nu1": 0.005,  "nu2": 0.005,
                  "epsilon": 0.0, "kappa": 0.0}

        r_low  = run_single(params=p_low)
        r_high = run_single(params=p_high)

        peak_low  = self._peak_I_total(r_low)
        peak_high = self._peak_I_total(r_high)

        assert peak_high < peak_low, (
            f"Higher vaccination should reduce peak. "
            f"Low vax peak = {peak_low:.0f}, high vax peak = {peak_high:.0f}."
        )

    def test_higher_efficacy_decreases_r0(self):
        """Higher vaccine efficacy → lower R0 (fewer susceptibles at DFE)."""
        p_low  = {**DEFAULT_PARAMS, "delta": 0.3}
        p_high = {**DEFAULT_PARAMS, "delta": 0.95}

        R0_low  = compute_r0(p_low)
        R0_high = compute_r0(p_high)

        assert R0_high < R0_low, (
            f"Higher delta should reduce R0. "
            f"delta=0.3 → R0={R0_low:.4f}, delta=0.95 → R0={R0_high:.4f}."
        )

    def test_higher_kappa_decreases_peak(self):
        """
        Higher behavioral response → more contact reduction at peak
        → lower maximum infectious count.
        Kappa does not affect R0, but it does affect peak size.
        """
        p_low  = {**DEFAULT_PARAMS, "kappa": 0.0,  "epsilon": 0.0}
        p_high = {**DEFAULT_PARAMS, "kappa": 15.0, "epsilon": 0.0}

        r_low  = run_single(params=p_low)
        r_high = run_single(params=p_high)

        peak_low  = self._peak_I_total(r_low)
        peak_high = self._peak_I_total(r_high)

        assert peak_high < peak_low, (
            f"Higher kappa should reduce peak. "
            f"kappa=0 peak = {peak_low:.0f}, kappa=15 peak = {peak_high:.0f}."
        )

    def test_higher_epsilon_increases_seasonal_variation(self):
        """
        Higher seasonality amplitude → larger difference between
        winter peak and summer trough in beta(t).
        """
        p_low  = DEFAULT_PARAMS.copy()
        p_high = DEFAULT_PARAMS.copy()
        p_low["epsilon"]  = 0.05
        p_high["epsilon"] = 0.40

        # Beta at winter peak (t=0) vs summer trough (t=182)
        def seasonal_variation(p):
            return seasonal_beta(0, p) - seasonal_beta(182, p)

        var_low  = seasonal_variation(p_low)
        var_high = seasonal_variation(p_high)

        assert var_high > var_low, (
            f"Higher epsilon should increase seasonal variation. "
            f"epsilon=0.05 variation={var_low:.5f}, "
            f"epsilon=0.40 variation={var_high:.5f}."
        )

    def test_higher_omega_produces_recurrent_waves(self):
        """
        Faster immunity waning → more people returning to susceptible
        → stronger recurrent epidemic waves in later years.

        We measure this by comparing the ratio of the second-year
        peak to the first-year peak. With fast waning, the second
        wave should be relatively large. With slow waning (omega≈0),
        most people remain immune and the second wave is negligible.
        """
        p_slow_waning = {**DEFAULT_PARAMS, "omega": 1/1000, "epsilon": 0.0,
                         "kappa": 0.0}
        p_fast_waning = {**DEFAULT_PARAMS, "omega": 1/60,   "epsilon": 0.0,
                         "kappa": 0.0}

        r_slow = run_single(params=p_slow_waning)
        r_fast = run_single(params=p_fast_waning)

        # Second year: days 365-730 (indices 365 to 729)
        I_slow = r_slow.y[:, 2] + r_slow.y[:, 7]
        I_fast = r_fast.y[:, 2] + r_fast.y[:, 7]

        second_year_peak_slow = I_slow[365:730].max()
        second_year_peak_fast = I_fast[365:730].max()

        assert second_year_peak_fast > second_year_peak_slow, (
            f"Faster immunity waning should produce larger second-year wave. "
            f"Slow waning second peak = {second_year_peak_slow:.0f}, "
            f"Fast waning second peak = {second_year_peak_fast:.0f}."
        )


# =============================================================================
# SAMPLER TESTS
# =============================================================================

class TestSampler:
    """Validates the Latin Hypercube Sampling and validity filter."""

    def test_single_sample_has_all_keys(self):
        """A single sample must contain all required parameter keys."""
        rng    = np.random.default_rng(seed=0)
        sample = draw_single_sample(rng)

        required_keys = list(DEFAULT_PARAMS.keys())
        for key in required_keys:
            assert key in sample, f"Key '{key}' missing from single sample."

    def test_single_sample_within_bounds(self):
        """Every sampled parameter must fall within PARAM_BOUNDS."""
        rng = np.random.default_rng(seed=0)
        for _ in range(20):
            sample = draw_single_sample(rng)
            for param, (lo, hi) in PARAM_BOUNDS.items():
                assert lo <= sample[param] <= hi, (
                    f"Parameter {param} = {sample[param]:.6f} outside "
                    f"bounds [{lo:.6f}, {hi:.6f}]."
                )

    def test_sigma_gamma_ratio_within_bounds(self):
        """The sigma/gamma ratio must be biologically plausible."""
        rng = np.random.default_rng(seed=0)
        lo_r, hi_r = SIGMA_GAMMA_RATIO_BOUNDS
        for _ in range(20):
            sample = draw_single_sample(rng)
            ratio  = sample["sigma"] / sample["gamma"]
            assert lo_r <= ratio <= hi_r, (
                f"sigma/gamma = {ratio:.3f} outside bounds [{lo_r}, {hi_r}]. "
                f"sigma={sample['sigma']:.5f}, gamma={sample['gamma']:.5f}."
            )

    def test_ensemble_correct_length(self):
        """draw_ensemble must return exactly n parameter dicts."""
        n   = 30
        ens = draw_ensemble(n=n, seed=42)
        assert len(ens) == n, (
            f"Expected {n} samples, got {len(ens)}."
        )

    def test_ensemble_reproducibility(self):
        """Same seed must produce identical ensemble."""
        ens_a = draw_ensemble(n=20, seed=42)
        ens_b = draw_ensemble(n=20, seed=42)
        for i in range(20):
            assert abs(ens_a[i]["beta0"] - ens_b[i]["beta0"]) < 1e-12, (
                f"Ensemble not reproducible. "
                f"Run {i}: beta0_a={ens_a[i]['beta0']}, beta0_b={ens_b[i]['beta0']}."
            )

    def test_ensemble_different_seeds_differ(self):
        """Different seeds must produce different ensembles."""
        ens_a = draw_ensemble(n=20, seed=42)
        ens_b = draw_ensemble(n=20, seed=99)
        any_different = any(
            abs(ens_a[i]["beta0"] - ens_b[i]["beta0"]) > 1e-8
            for i in range(20)
        )
        assert any_different, (
            "Different seeds produced identical ensembles — seed is not working."
        )

    def test_ensemble_fixed_params_unchanged(self):
        """Fixed parameters must not be altered by the sampler."""
        ens = draw_ensemble(n=10, seed=42)
        for i, sample in enumerate(ens):
            for key, val in FIXED_PARAMS.items():
                assert abs(sample[key] - val) < 1e-12, (
                    f"Fixed parameter {key} was changed in sample {i}. "
                    f"Expected {val}, got {sample[key]}."
                )


# =============================================================================
# SOLVER TESTS
# =============================================================================

class TestSolver:
    """Validates solver output structure and basic properties."""

    def test_single_run_success(self, default_single_run):
        """Solver must complete successfully with default params."""
        assert default_single_run.success, (
            f"Solver failed: {default_single_run.message}"
        )

    def test_single_run_output_shape(self, default_single_run):
        """Output array must have correct shape (N_timepoints, 10)."""
        expected_shape = (len(T_EVAL), 10)
        actual_shape   = default_single_run.y.shape
        assert actual_shape == expected_shape, (
            f"Expected shape {expected_shape}, got {actual_shape}."
        )

    def test_all_compartments_non_negative(self, default_single_run):
        """
        Compartment values must never go negative.
        A negative susceptible count is physically meaningless and
        indicates a numerical instability or step-size issue.
        """
        min_val = default_single_run.y.min()
        assert min_val >= -1.0, (
            f"Compartment went negative (min = {min_val:.4f}). "
            f"This indicates numerical instability — try reducing max_step."
        )

    def test_ensemble_output_shape(self, small_ensemble):
        """Ensemble result must have shape (N_runs, N_timepoints, 10)."""
        ens      = small_ensemble
        n_runs   = ens.trajectories.shape[0]
        expected = (n_runs, len(T_EVAL), 10)
        assert ens.trajectories.shape == expected, (
            f"Expected ensemble shape {expected}, got {ens.trajectories.shape}."
        )

    def test_percentile_ordering(self, small_ensemble):
        """Lower CI must be <= median <= upper CI at all times."""
        ens = small_ensemble
        assert np.all(ens.lower <= ens.median + 1e-6), (
            "Lower CI exceeds median at some timepoint."
        )
        assert np.all(ens.upper >= ens.median - 1e-6), (
            "Upper CI is below median at some timepoint."
        )

    def test_no_failed_runs(self, small_ensemble):
        """All ODE runs should converge with the calibrated parameters."""
        assert small_ensemble.n_failed == 0, (
            f"{small_ensemble.n_failed} ensemble runs failed to converge. "
            f"Check parameter bounds in parameters.py."
        )


# =============================================================================
# ANALYSIS TESTS
# =============================================================================

class TestAnalysis:
    """Validates R0 computation, DFE, and statistical outputs."""

    def test_dfe_sums_to_population(self):
        """S* + V* must equal N_i for each age group at the DFE."""
        dfe = compute_dfe(DEFAULT_PARAMS)
        N1, N2 = POPULATION["N1"], POPULATION["N2"]

        err1 = abs(dfe["S1_star"] + dfe["V1_star"] - N1)
        err2 = abs(dfe["S2_star"] + dfe["V2_star"] - N2)

        assert err1 < 1.0, (
            f"S1* + V1* = {dfe['S1_star']+dfe['V1_star']:.1f}, expected {N1}. "
            f"Error = {err1:.2f}. Check compute_dfe() algebra."
        )
        assert err2 < 1.0, (
            f"S2* + V2* = {dfe['S2_star']+dfe['V2_star']:.1f}, expected {N2}. "
            f"Error = {err2:.2f}. Check compute_dfe() algebra."
        )

    def test_r0_positive(self):
        """R0 must always be a positive number."""
        R0 = compute_r0(DEFAULT_PARAMS)
        assert R0 > 0, f"R0 = {R0:.4f} is not positive."

    def test_r0_ensemble_all_positive(self, small_param_list):
        """All ensemble R0 values must be positive."""
        r0_vals = compute_r0_ensemble(small_param_list)
        assert np.all(r0_vals > 0), (
            f"Some R0 values are non-positive. Min = {r0_vals.min():.4f}."
        )

    def test_r0_ensemble_length(self, small_param_list):
        """R0 ensemble must have one value per parameter set."""
        r0_vals = compute_r0_ensemble(small_param_list)
        assert len(r0_vals) == len(small_param_list), (
            f"Expected {len(small_param_list)} R0 values, got {len(r0_vals)}."
        )

    def test_peak_stats_positive(self, default_single_run):
        """All peak statistics must be positive values."""
        peak = compute_peak_stats(default_single_run.y, default_single_run.t)
        for key, val in peak.items():
            assert val >= 0, (
                f"Peak stat '{key}' = {val:.4f} is negative."
            )

    def test_final_size_between_zero_and_one(self, default_single_run):
        """Final size fractions must be between 0 and 1."""
        fsize = compute_final_size(default_single_run.y)
        for key, val in fsize.items():
            assert 0.0 <= val <= 1.0, (
                f"Final size '{key}' = {val:.4f} outside [0, 1]."
            )

    def test_phi_equals_one_when_no_disease(self):
        """
        Behavioral factor Phi must equal exactly 1 when I_total = 0.
        At the start of an epidemic (or at the DFE), there is no
        disease and no behavioral change.
        """
        N   = POPULATION["N"]
        phi = behavioral_factor(0.0, N, DEFAULT_PARAMS)
        assert abs(phi - 1.0) < 1e-12, (
            f"Phi = {phi:.8f} when I_total=0, expected 1.0."
        )

    def test_phi_decreases_with_prevalence(self):
        """Phi must strictly decrease as prevalence rises."""
        N      = POPULATION["N"]
        prev   = [0.0, 0.01, 0.05, 0.10, 0.20]
        phis   = [behavioral_factor(p * N, N, DEFAULT_PARAMS) for p in prev]

        for i in range(len(phis) - 1):
            assert phis[i] > phis[i+1], (
                f"Phi did not decrease from prevalence {prev[i]*100:.0f}% "
                f"to {prev[i+1]*100:.0f}%. "
                f"Phi values: {[f'{p:.4f}' for p in phis]}."
            )

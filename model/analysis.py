"""
analysis.py
===========
Extracts epidemiological meaning from solver output.

WHAT THIS FILE DOES
-------------------
Takes the structured results from solver.py and computes the quantities
that actually answer the scientific questions:

    - R0: the basic reproduction number, computed analytically via the
      next-generation matrix (NGM). Returns a distribution of R0 values
      across ensemble members.

    - Peak statistics: when does the epidemic peak, how large is it,
      and how does it differ between age groups?

    - Final size: what fraction of each group was infected over 5 years?

    - Sensitivity: which parameters drive the most uncertainty in R0
      and peak infectious count? (Ranked by Spearman correlation.)

    - Disease-free equilibrium: the S*, V* values needed for the NGM.

THE NEXT-GENERATION MATRIX AND R0
-----------------------------------
R0 is the average number of secondary infections caused by one
infectious person introduced into a fully susceptible population.
For a multi-group model, it cannot be read off the equations directly —
it requires the next-generation matrix (NGM) method.

The NGM approach splits the linearized system at the disease-free
equilibrium (DFE) into two matrices:

    F: the rate of NEW infections entering each infected compartment.
       Only contains terms of the form "transmission event creates
       a new exposed person."

    V: the rate of TRANSITIONS out of each infected compartment by
       any process other than new infection (progression, recovery, death).

The NGM is K = F * V_inverse. R0 = dominant eigenvalue (spectral
radius) of K.

For our model, the infected compartments are [E1, E2, I1, I2].
After working through the algebra (see project report Section 4),
K collapses to a clean 2x2 matrix:

    K[i,j] = beta0 * c_ij * sigma / ((sigma+mu) * (gamma+mu))
             * (S*_i + (1-delta) * V*_i) / N_j

where S*_i and V*_i are the DFE values.

R0 = dominant eigenvalue of this 2x2 K matrix
   = (K11 + K22)/2 + sqrt(((K11-K22)/2)^2 + K12*K21)

CRITICAL NOTE ON KAPPA AND R0
------------------------------
The behavioral response parameter kappa does NOT appear in R0.
R0 is evaluated at the disease-free equilibrium, where I_total ~ 0.
This makes the behavioral dampening factor Phi = 1/(1 + kappa*0/N) = 1
exactly — behavioral effects vanish when there is almost no disease.

Therefore: kappa does not change R0. It only changes the epidemic's
trajectory after it takes off. Two simulations with identical R0 but
different kappa values will have the same initial doubling time but
very different peak sizes and wave structures.
This is one of the analytically interesting results the project demonstrates.
"""

import numpy as np
from scipy import stats as sp_stats

from model.parameters import (
    DEFAULT_PARAMS,
    FIXED_PARAMS,
    POPULATION,
    SAMPLED_PARAM_NAMES,
)
from model.sampler import ensemble_to_array


# =============================================================================
# DISEASE-FREE EQUILIBRIUM
# =============================================================================

def compute_dfe(params: dict) -> dict:
    """
    Computes the disease-free equilibrium (DFE) compartment values.

    At the DFE, E=I=0 for all groups. The S, R, V subsystem reaches
    a steady state determined by births, deaths, vaccination, and
    waning. Since R=0 when I=0 (no recoveries without infections),
    the DFE simplifies to just S* and V* for each group.

    Derivation (for each group i, dropping subscripts):
        At equilibrium: dS/dt = 0, dV/dt = 0, R=0
        dS/dt = mu*Ni + omega_v*V - nu*S - mu*S = 0
        dV/dt = nu*S - omega_v*V - mu*V = 0

        From dV/dt: nu*S = (omega_v + mu)*V
                    V = nu*S / (omega_v + mu)

        Substituting into dS/dt:
            mu*Ni + omega_v * nu*S/(omega_v+mu) - nu*S - mu*S = 0
            Solving for S*...

    Parameters
    ----------
    params : dict
        Complete parameter dictionary.

    Returns
    -------
    dict with keys:
        'S1_star' : DFE susceptible youth
        'V1_star' : DFE vaccinated youth
        'S2_star' : DFE susceptible adults
        'V2_star' : DFE vaccinated adults
    """
    mu      = params["mu"]
    omega_v = params["omega_v"]
    nu1     = params["nu1"]
    nu2     = params["nu2"]
    N1      = POPULATION["N1"]
    N2      = POPULATION["N2"]

    def _dfe_group(N, nu):
        """
        Solves for S* and V* for one age group.
        From dS/dt=0 and dV/dt=0 at R=0, E=0, I=0:

            S* = mu*N*(omega_v + mu) / [(nu + mu)*(omega_v + mu) - nu*omega_v]
                 (after algebra — simplifies because omega_v*nu terms appear)

            Equivalently, using the denominator D:
            D  = mu*(omega_v + mu) + nu*(omega_v + mu) - nu*omega_v
               = mu*(omega_v + mu) + nu*mu
            S* = mu*N*(omega_v + mu) / D
            V* = nu*S* / (omega_v + mu)
        """
        if mu == 0.0:
            # No vital dynamics: steady state is S*=N*(omega_v/(omega_v+nu)),
            # V*=N*(nu/(omega_v+nu)). When nu=0 as well, S*=N, V*=0.
            denom = omega_v + nu
            if denom == 0.0:
                return float(N), 0.0
            return float(N) * omega_v / denom, float(N) * nu / denom
        D      = mu * (omega_v + mu) + nu * mu
        S_star = mu * N * (omega_v + mu) / D
        V_star = nu * S_star / (omega_v + mu)
        return S_star, V_star

    S1_star, V1_star = _dfe_group(N1, nu1)
    S2_star, V2_star = _dfe_group(N2, nu2)

    return {
        "S1_star": S1_star,
        "V1_star": V1_star,
        "S2_star": S2_star,
        "V2_star": V2_star,
    }


# =============================================================================
# NEXT-GENERATION MATRIX AND R0
# =============================================================================

def compute_ngm(params: dict, dfe: dict) -> np.ndarray:
    """
    Computes the 2x2 next-generation matrix K.

    Each entry K[i,j] = expected number of secondary infections in
    group i caused by one infectious person in group j, in a fully
    susceptible population at the DFE.

    K[i,j] = beta0 * c_ij * sigma / ((sigma+mu)*(gamma+mu))
              * (S*_i + (1-delta)*V*_i) / N_j

    Breaking this down:
        beta0 * c_ij              — transmission rate from j to i
        sigma / (sigma + mu)      — probability an exposed person
                                    survives to become infectious
                                    (vs dying during latency)
        1 / (gamma + mu)          — average duration of infectiousness
        S*_i + (1-delta)*V*_i     — effective susceptibles in group i
                                    (unvaccinated plus partially
                                    protected vaccinated people)
        / N_j                     — normalizes by group j size

    Parameters
    ----------
    params : dict
        Complete parameter set.
    dfe : dict
        Disease-free equilibrium values from compute_dfe().

    Returns
    -------
    np.ndarray, shape (2, 2)
        The next-generation matrix K.
    """
    beta0  = params["beta0"]
    sigma  = params["sigma"]
    gamma  = params["gamma"]
    mu     = params["mu"]
    delta  = params["delta"]
    c11    = params["c11"]
    c12    = params["c12"]
    c21    = params["c21"]
    c22    = params["c22"]

    N1 = POPULATION["N1"]
    N2 = POPULATION["N2"]

    # Effective susceptibles in each group at DFE
    # Unvaccinated susceptibles + vaccinated people who can still be infected
    eff1 = dfe["S1_star"] + (1 - delta) * dfe["V1_star"]
    eff2 = dfe["S2_star"] + (1 - delta) * dfe["V2_star"]

    # Factor shared by all K entries:
    # transmission per contact * probability E survives to I * duration of I
    # sigma/(sigma+mu) ~ 1 (most exposed people reach I before dying)
    # 1/(gamma+mu) ~ 1/gamma = 10 days (average infectious period)
    common = beta0 * (sigma / (sigma + mu)) * (1.0 / (gamma + mu))

    # K[i,j]: infections in group i from one infectious in group j
    K = np.array([
        [common * c11 * eff1 / N1,   common * c12 * eff1 / N2],
        [common * c21 * eff2 / N1,   common * c22 * eff2 / N2],
    ])

    return K


def compute_r0(params: dict) -> float:
    """
    Computes R0 as the dominant eigenvalue of the NGM.

    For a 2x2 matrix, the dominant eigenvalue has a closed-form:
        R0 = (K11 + K22)/2 + sqrt(((K11-K22)/2)^2 + K12*K21)

    We compute it both ways (eigenvalue and formula) and verify they
    agree — this serves as an internal consistency check.

    Parameters
    ----------
    params : dict
        Complete parameter set.

    Returns
    -------
    float
        The basic reproduction number R0.
        R0 > 1: epidemic can establish.
        R0 < 1: epidemic dies out.
        R0 = 1: threshold (herd immunity boundary).

    Notes
    -----
    kappa is NOT in this calculation. See module docstring for
    the detailed explanation of why behavioral adaptation does
    not affect R0.

    beta0 (not beta(t)) is used because R0 is a time-averaged
    quantity evaluated at the DFE. Using the seasonal beta(t)
    would make R0 depend on what time of year the epidemic starts,
    which is a different (and more complex) question.
    """
    dfe = compute_dfe(params)
    K   = compute_ngm(params, dfe)

    # Method 1: closed-form formula for 2x2 dominant eigenvalue
    K11, K12 = K[0, 0], K[0, 1]
    K21, K22 = K[1, 0], K[1, 1]
    discriminant = ((K11 - K22) / 2.0) ** 2 + K12 * K21
    R0_formula   = (K11 + K22) / 2.0 + np.sqrt(max(discriminant, 0.0))

    # Method 2: numpy eigenvalue (dominant = largest real part)
    eigenvalues = np.linalg.eigvals(K)
    R0_numeric  = float(np.max(eigenvalues.real))

    # Internal consistency check (tolerance for floating-point differences)
    if abs(R0_formula - R0_numeric) > 1e-6:
        print(f"  WARNING: R0 formula ({R0_formula:.6f}) and "
              f"eigenvalue ({R0_numeric:.6f}) disagree by "
              f"{abs(R0_formula - R0_numeric):.2e}")

    return R0_formula


def compute_r0_ensemble(param_list: list) -> np.ndarray:
    """
    Computes R0 for every member of the ensemble.

    Parameters
    ----------
    param_list : list of dict
        Output of draw_ensemble() from sampler.py.

    Returns
    -------
    np.ndarray, shape (N,)
        One R0 value per ensemble member.
        This array is a probability distribution over R0 values,
        reflecting our uncertainty about the true R0.
    """
    return np.array([compute_r0(p) for p in param_list])


# =============================================================================
# PEAK STATISTICS
# =============================================================================

def compute_peak_stats(result_y: np.ndarray, t: np.ndarray) -> dict:
    """
    Computes peak epidemic statistics from a single trajectory.

    Parameters
    ----------
    result_y : np.ndarray, shape (N_timepoints, 10)
        Compartment values from SingleResult.y or one row of
        EnsembleResult.trajectories.
    t : np.ndarray, shape (N_timepoints,)
        Time in days.

    Returns
    -------
    dict with keys:
        'peak_I_total'     : maximum total infectious count
        'peak_day'         : day of peak total infectious
        'peak_year'        : peak day / 365
        'peak_I1'          : peak infectious youth count
        'peak_I2'          : peak infectious adults count
        'peak_prevalence'  : peak I_total / N as percentage
    """
    I1 = result_y[:, 2]
    I2 = result_y[:, 7]
    I_total = I1 + I2
    N = POPULATION["N"]

    peak_idx = I_total.argmax()

    return {
        "peak_I_total"    : float(I_total[peak_idx]),
        "peak_day"        : float(t[peak_idx]),
        "peak_year"       : float(t[peak_idx]) / 365.0,
        "peak_I1"         : float(I1[peak_idx]),
        "peak_I2"         : float(I2[peak_idx]),
        "peak_prevalence" : 100.0 * float(I_total[peak_idx]) / N,
    }


def compute_ensemble_peak_stats(ensemble_result) -> dict:
    """
    Computes peak statistics across all ensemble members.

    Parameters
    ----------
    ensemble_result : EnsembleResult
        Output of solver.run_ensemble().

    Returns
    -------
    dict with keys for each peak stat:
        '{stat}_median', '{stat}_lower', '{stat}_upper'
        where lower = 2.5th percentile, upper = 97.5th percentile.
    """
    stats_list = [
        compute_peak_stats(ensemble_result.trajectories[i],
                           ensemble_result.t)
        for i in range(ensemble_result.trajectories.shape[0])
    ]

    keys = stats_list[0].keys()
    result = {}
    for key in keys:
        values = np.array([s[key] for s in stats_list])
        result[f"{key}_median"] = float(np.percentile(values, 50))
        result[f"{key}_lower"]  = float(np.percentile(values,  2.5))
        result[f"{key}_upper"]  = float(np.percentile(values, 97.5))

    return result


# =============================================================================
# FINAL SIZE
# =============================================================================

def compute_final_size(result_y: np.ndarray) -> dict:
    """
    Computes the cumulative infection burden over the simulation.

    Final size = fraction of each group that passed through I at
    least once. Approximated as 1 - S(end)/N_group, which slightly
    underestimates because some people may be in E or I at t=end.

    Parameters
    ----------
    result_y : np.ndarray, shape (N_timepoints, 10)

    Returns
    -------
    dict:
        'final_size_youth'  : fraction of youth infected over 5 years
        'final_size_adults' : fraction of adults infected
        'final_size_total'  : fraction of total population infected
    """
    N1 = POPULATION["N1"]
    N2 = POPULATION["N2"]
    N  = POPULATION["N"]

    S1_end = result_y[-1, 0]
    S2_end = result_y[-1, 5]

    return {
        "final_size_youth"  : 1.0 - S1_end / N1,
        "final_size_adults" : 1.0 - S2_end / N2,
        "final_size_total"  : 1.0 - (S1_end + S2_end) / N,
    }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def compute_sensitivity(
    param_list: list,
    r0_values: np.ndarray,
    ensemble_result,
) -> dict:
    """
    Ranks parameters by their influence on R0 and peak infectious count
    using Spearman rank correlation.

    Spearman correlation is used rather than Pearson because the
    relationships between parameters and outputs may be nonlinear.
    Spearman measures monotonic association — if R0 consistently
    increases as beta0 increases (regardless of the exact shape),
    the Spearman correlation will be high.

    Parameters
    ----------
    param_list : list of dict
        Ensemble parameter sets from draw_ensemble().
    r0_values : np.ndarray, shape (N,)
        R0 for each ensemble member from compute_r0_ensemble().
    ensemble_result : EnsembleResult
        Solver output from run_ensemble().

    Returns
    -------
    dict with keys:
        'r0_sensitivity'   : list of (param_name, spearman_rho) tuples,
                             sorted by |rho| descending
        'peak_sensitivity' : same but for peak I_total
    """
    param_array = ensemble_to_array(param_list)
    # param_array shape: (N, D) where D = number of sampled params

    # Peak I_total for each ensemble member
    peak_I = np.array([
        (ensemble_result.trajectories[i, :, 2] +
         ensemble_result.trajectories[i, :, 7]).max()
        for i in range(ensemble_result.trajectories.shape[0])
    ])

    r0_sensitivity   = []
    peak_sensitivity = []

    for j, name in enumerate(SAMPLED_PARAM_NAMES):
        col = param_array[:, j]

        # Spearman correlation with R0
        rho_r0, _ = sp_stats.spearmanr(col, r0_values)
        r0_sensitivity.append((name, float(rho_r0)))

        # Spearman correlation with peak infectious
        rho_peak, _ = sp_stats.spearmanr(col, peak_I)
        peak_sensitivity.append((name, float(rho_peak)))

    # Sort by absolute correlation magnitude (most influential first)
    r0_sensitivity.sort(key=lambda x: abs(x[1]), reverse=True)
    peak_sensitivity.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "r0_sensitivity"   : r0_sensitivity,
        "peak_sensitivity" : peak_sensitivity,
    }


# =============================================================================
# SELF-CHECK  (run with: python model/analysis.py)
# =============================================================================

if __name__ == "__main__":
    from model.solver import run_single, run_ensemble

    print("=" * 62)
    print("ANALYSIS.PY — SELF-CHECK")
    print("=" * 62)

    p = DEFAULT_PARAMS

    # ------------------------------------------------------------------
    print("\n[1] Disease-free equilibrium (default params)")
    dfe = compute_dfe(p)
    N1, N2, N = POPULATION["N1"], POPULATION["N2"], POPULATION["N"]
    print(f"    S1* = {dfe['S1_star']:>12,.1f}  (of N1={N1:,})")
    print(f"    V1* = {dfe['V1_star']:>12,.1f}")
    print(f"    S1*+V1* = {dfe['S1_star']+dfe['V1_star']:>10,.1f}  "
          f"(should equal N1={N1:,})")
    print(f"    S2* = {dfe['S2_star']:>12,.1f}  (of N2={N2:,})")
    print(f"    V2* = {dfe['V2_star']:>12,.1f}")
    print(f"    S2*+V2* = {dfe['S2_star']+dfe['V2_star']:>10,.1f}  "
          f"(should equal N2={N2:,})")

    # ------------------------------------------------------------------
    print("\n[2] NGM and R0 (default params)")
    K   = compute_ngm(p, dfe)
    R0  = compute_r0(p)
    print(f"    NGM K:")
    print(f"      [ {K[0,0]:.4f}  {K[0,1]:.4f} ]")
    print(f"      [ {K[1,0]:.4f}  {K[1,1]:.4f} ]")
    print(f"    R0 = {R0:.4f}")
    print(f"    Epidemic will {'grow' if R0>1 else 'die out'}  "
          f"(R0 {'>' if R0>1 else '<'} 1)")

    # ------------------------------------------------------------------
    print("\n[3] Kappa independence check")
    # R0 should be identical regardless of kappa value
    p_low_kappa  = {**p, "kappa": 0.0}
    p_high_kappa = {**p, "kappa": 50.0}
    R0_low  = compute_r0(p_low_kappa)
    R0_high = compute_r0(p_high_kappa)
    print(f"    R0 with kappa=0:   {R0_low:.6f}")
    print(f"    R0 with kappa=5:   {R0:.6f}")
    print(f"    R0 with kappa=50:  {R0_high:.6f}")
    kappa_independent = abs(R0_low - R0_high) < 1e-10
    print(f"    R0 independent of kappa: {kappa_independent}  "
          f"{'PASS' if kappa_independent else 'FAIL'}")

    # ------------------------------------------------------------------
    print("\n[4] Threshold check — R0 < 1 with high vaccination")
    p_high_vax = {**p, "nu1": 0.01, "nu2": 0.01, "delta": 0.99}
    R0_high_vax = compute_r0(p_high_vax)
    print(f"    R0 with high vaccination: {R0_high_vax:.4f}")
    print(f"    Below threshold:          {R0_high_vax < 1.0}  "
          f"{'PASS' if R0_high_vax < 1.0 else 'try higher nu'}")

    # ------------------------------------------------------------------
    print("\n[5] Peak statistics (single run)")
    single = run_single()
    peak = compute_peak_stats(single.y, single.t)
    for k, v in peak.items():
        print(f"    {k:<22} = {v:.2f}")

    # ------------------------------------------------------------------
    print("\n[6] Final size (single run)")
    fs = compute_final_size(single.y)
    for k, v in fs.items():
        print(f"    {k:<25} = {v*100:.1f}%")

    # ------------------------------------------------------------------
    print("\n[7] Small ensemble R0 distribution (n=30)")
    from model.sampler import draw_ensemble
    ens_params = draw_ensemble(n=30, seed=42)
    r0_vals    = compute_r0_ensemble(ens_params)
    print(f"    R0 across 30 ensemble members:")
    print(f"      min    = {r0_vals.min():.4f}")
    print(f"      median = {np.median(r0_vals):.4f}")
    print(f"      mean   = {r0_vals.mean():.4f}")
    print(f"      max    = {r0_vals.max():.4f}")
    print(f"      std    = {r0_vals.std():.4f}")
    print(f"    All R0 > 1: {np.all(r0_vals > 1.0)}")

    # ------------------------------------------------------------------
    print("\n[8] Sensitivity analysis (n=30 ensemble)")
    ens_result = run_ensemble(n=30, seed=42, verbose=False)
    sens = compute_sensitivity(ens_params, r0_vals, ens_result)

    print(f"    R0 sensitivity (Spearman rho, ranked):")
    for name, rho in sens["r0_sensitivity"]:
        bar = "+" * int(abs(rho) * 20)
        direction = "pos" if rho > 0 else "neg"
        print(f"      {name:<12} rho={rho:+.3f}  [{direction}]  {bar}")

    print(f"\n    Peak infectious sensitivity (Spearman rho, ranked):")
    for name, rho in sens["peak_sensitivity"]:
        bar = "+" * int(abs(rho) * 20)
        direction = "pos" if rho > 0 else "neg"
        print(f"      {name:<12} rho={rho:+.3f}  [{direction}]  {bar}")

    print("\n" + "=" * 62)
    print("analysis.py self-check complete.")
    print("=" * 62)

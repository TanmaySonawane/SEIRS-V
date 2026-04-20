"""
equations.py
============
The mathematical heart of the model.

This file defines the right-hand side of the ODE system — the function
that scipy's solver calls thousands of times to compute the trajectory
of the epidemic.

WHAT THIS FILE DOES
-------------------
It takes the current state of the population (how many people are in
each compartment right now) and returns the rate of change of every
compartment (how fast each one is growing or shrinking at this instant).

That one function — seirs_v_odes() — IS the model. Everything else in
the project either feeds inputs into it or processes its outputs.

HOW scipy USES THIS FUNCTION
-----------------------------
scipy's solve_ivp works like this:

    1. Start at t=0 with initial conditions y0
    2. Call seirs_v_odes(t=0, y=y0, params=...) → get dy/dt
    3. Take a small step forward in time using those rates
    4. Repeat from the new position, adjusting step size for accuracy
    5. Continue until t_end is reached

It calls our function potentially thousands of times per simulation.
That is why this function must be fast and correct — no print statements,
no slow operations, pure numpy math only.

STATE VECTOR INDEX REFERENCE  (memorize or bookmark this)
----------------------------------------------------------
y[0] = S1   Susceptible youth
y[1] = E1   Exposed youth
y[2] = I1   Infectious youth
y[3] = R1   Recovered youth
y[4] = V1   Vaccinated youth

y[5] = S2   Susceptible adults
y[6] = E2   Exposed adults
y[7] = I2   Infectious adults
y[8] = R2   Recovered adults
y[9] = V2   Vaccinated adults
"""

import numpy as np
from model.parameters import POPULATION, get_contact_matrix


# =============================================================================
# INTERMEDIATE COMPUTATIONS
# These are the building blocks that get assembled into the ODEs.
# Each is its own function so it can be tested and understood in isolation.
# =============================================================================

def seasonal_beta(t: float, params: dict) -> float:
    """
    Computes the time-varying transmission rate β(t).

    β(t) = β₀ × (1 + ε × cos(2π t / 365))

    Parameters
    ----------
    t : float
        Current time in days.
    params : dict
        Must contain 'beta0' and 'epsilon'.

    Returns
    -------
    float
        Transmission probability per contact at time t.

    Notes
    -----
    At t = 0   (start of simulation, assumed winter): cos(0) = +1
                β = β₀ × (1 + ε)   ← maximum transmission
    At t = 182 (summer):             cos(π) = -1
                β = β₀ × (1 - ε)   ← minimum transmission
    At t = 365 (next winter):        cos(2π) = +1
                β is back at maximum — one full seasonal cycle complete.

    If you want to test the model WITHOUT seasonality:
        set params['epsilon'] = 0
        This makes cos(...) irrelevant and β(t) = β₀ always.
    """
    beta0   = params["beta0"]
    epsilon = params["epsilon"]

    return beta0 * (1.0 + epsilon * np.cos(2.0 * np.pi * t / 365.0))


def behavioral_factor(I_total: float, N: float, params: dict) -> float:
    """
    Computes the behavioral dampening factor Φ(t).

    Φ(t) = 1 / (1 + κ × I_total / N)

    This represents the population-level reduction in contact rates
    as people observe rising prevalence and voluntarily reduce mixing.

    Parameters
    ----------
    I_total : float
        Total number of infectious individuals across BOTH age groups
        at the current moment: I1 + I2.
    N : float
        Total population size (constant).
    params : dict
        Must contain 'kappa'.

    Returns
    -------
    float
        A value between 0 and 1.
        Φ = 1.0  → no behavioral change (epidemic is small or kappa=0)
        Φ → 0    → strong contact reduction (high prevalence or high kappa)

    Examples
    --------
    kappa=5, prevalence=1%:   Φ = 1/(1 + 5×0.01) = 1/1.05 ≈ 0.952
    kappa=5, prevalence=5%:   Φ = 1/(1 + 5×0.05) = 1/1.25 = 0.800
    kappa=5, prevalence=10%:  Φ = 1/(1 + 5×0.10) = 1/1.50 ≈ 0.667

    Notes
    -----
    If you want to test WITHOUT behavioral adaptation:
        set params['kappa'] = 0
        Φ = 1/(1+0) = 1.0 always — no dampening.
    """
    kappa = params["kappa"]

    # Prevalence: what fraction of the total population is infectious now
    prevalence = I_total / N

    return 1.0 / (1.0 + kappa * prevalence)


def force_of_infection(
    I1: float,
    I2: float,
    beta_t: float,
    phi: float,
    params: dict
) -> tuple:
    """
    Computes λ₁(t) and λ₂(t) — the force of infection on each age group.

    λᵢ(t) = β(t) × Φ(t) × Σⱼ [ cᵢⱼ × Iⱼ / Nⱼ ]

    This is the rate at which a single susceptible person in group i
    acquires infection per day.

    Parameters
    ----------
    I1 : float
        Current number of infectious youth (y[2]).
    I2 : float
        Current number of infectious adults (y[7]).
    beta_t : float
        Current seasonal transmission rate β(t) from seasonal_beta().
    phi : float
        Current behavioral dampening factor Φ(t) from behavioral_factor().
    params : dict
        Must contain contact matrix entries c11, c12, c21, c22.

    Returns
    -------
    tuple : (lambda1, lambda2)
        lambda1 : force of infection on youth
        lambda2 : force of infection on adults

    How the matrix multiplication works
    ------------------------------------
    The contact matrix C is:
        C = [[c11, c12],     [[youth-youth,  youth-adult ],
             [c21, c22]]      [adult-youth,  adult-adult ]]

    The infectious fraction vector is:
        f = [I1/N1, I2/N2]   (probability a random contact in each group
                               is currently infectious)

    The force of infection vector is:
        [λ1, λ2] = β(t) × Φ(t) × C @ f

    Where @ is numpy matrix multiplication.

    Written out explicitly:
        λ1 = β(t) × Φ(t) × (c11 × I1/N1 + c12 × I2/N2)
        λ2 = β(t) × Φ(t) × (c21 × I1/N1 + c22 × I2/N2)
    """
    N1 = POPULATION["N1"]
    N2 = POPULATION["N2"]

    # Build the contact matrix from current params
    # Shape: (2, 2)
    C = get_contact_matrix(params)

    # Infectious fraction in each group
    # f[0] = I1/N1 : probability that a random youth contact is infectious
    # f[1] = I2/N2 : probability that a random adult contact is infectious
    # Shape: (2,)
    f = np.array([I1 / N1, I2 / N2])

    # Matrix-vector multiplication: C @ f gives a (2,) array
    # result[0] = c11*(I1/N1) + c12*(I2/N2)  ← weighted contacts for youth
    # result[1] = c21*(I1/N1) + c22*(I2/N2)  ← weighted contacts for adults
    weighted_contacts = C @ f

    # Apply seasonal transmission rate and behavioral dampening
    # Both factors multiply the entire force of infection uniformly
    lambda1 = beta_t * phi * weighted_contacts[0]
    lambda2 = beta_t * phi * weighted_contacts[1]

    return lambda1, lambda2


# =============================================================================
# THE MAIN ODE FUNCTION
# This is the function scipy will call at every timestep.
# =============================================================================

def seirs_v_odes(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """
    Defines the right-hand side of the SEIRS-V ODE system.

    This function computes dy/dt — the rate of change of all 10
    compartments simultaneously at a given moment in time.

    scipy calls this function as:  dydt = seirs_v_odes(t, y, params)

    Parameters
    ----------
    t : float
        Current time in days. scipy passes this automatically.
        Even if your equations don't explicitly use t, scipy still
        passes it — our equations DO use it for seasonal forcing.

    y : np.ndarray, shape (10,)
        Current values of all compartments.
        y = [S1, E1, I1, R1, V1, S2, E2, I2, R2, V2]
        scipy updates this array at every step.

    params : dict
        All biological parameters. Passed via scipy's 'args' argument.

    Returns
    -------
    np.ndarray, shape (10,)
        The rate of change dy/dt for each compartment.
        Each element has units of "people per day."
        Positive = compartment is growing.
        Negative = compartment is shrinking.

    The 10 equations in order
    -------------------------
    dS1/dt = Λ1 + ω·R1 + ω_v·V1 - λ1·S1 - ν1·S1 - μ·S1
    dE1/dt = λ1·S1 + (1-δ)·λ1·V1 - σ·E1 - μ·E1
    dI1/dt = σ·E1 - γ·I1 - μ·I1
    dR1/dt = γ·I1 - ω·R1 - μ·R1
    dV1/dt = ν1·S1 - (1-δ)·λ1·V1 - ω_v·V1 - μ·V1

    dS2/dt = Λ2 + ω·R2 + ω_v·V2 - λ2·S2 - ν2·S2 - μ·S2
    dE2/dt = λ2·S2 + (1-δ)·λ2·V2 - σ·E2 - μ·E2
    dI2/dt = σ·E2 - γ·I2 - μ·I2
    dR2/dt = γ·I2 - ω·R2 - μ·R2
    dV2/dt = ν2·S2 - (1-δ)·λ2·V2 - ω_v·V2 - μ·V2
    """

    # -------------------------------------------------------------------------
    # STEP 1: Unpack the state vector
    # -------------------------------------------------------------------------
    # Give every compartment a readable name.
    # This is purely for clarity — y[0] and S1 are the same number in memory.
    # Never use raw index numbers like y[2] in the equations below —
    # a typo (y[3] instead of y[2]) would be nearly impossible to spot.

    S1, E1, I1, R1, V1 = y[0], y[1], y[2], y[3], y[4]
    S2, E2, I2, R2, V2 = y[5], y[6], y[7], y[8], y[9]

    # -------------------------------------------------------------------------
    # STEP 2: Unpack parameters
    # -------------------------------------------------------------------------
    # Pull every parameter out of the dict and name it clearly.
    # This avoids repeated dict lookups inside tight math and makes the
    # equations below read like the mathematical notation.

    sigma   = params["sigma"]      # E → I progression rate
    gamma   = params["gamma"]      # I → R recovery rate
    omega   = params["omega"]      # R → S natural immunity waning rate
    omega_v = params["omega_v"]    # V → S vaccine waning rate
    mu      = params["mu"]         # natural birth/death rate
    nu1     = params["nu1"]        # youth vaccination rate
    nu2     = params["nu2"]        # adult vaccination rate
    delta   = params["delta"]      # vaccine efficacy

    N1 = POPULATION["N1"]
    N2 = POPULATION["N2"]

    # Birth rates: set equal to death rate × group size
    # This keeps each age group's population constant over time.
    # Λ1 = μ × N1 means "births exactly replace deaths in the youth group"
    Lambda1 = mu * N1
    Lambda2 = mu * N2

    # -------------------------------------------------------------------------
    # STEP 3: Compute intermediate quantities
    # -------------------------------------------------------------------------
    # These are the building blocks computed ONCE per timestep,
    # then reused across multiple ODEs below.
    # Computing them once (rather than inside each equation) is both
    # faster and less error-prone.

    # Total infectious individuals across both groups right now
    I_total = I1 + I2

    # Seasonal transmission rate at this moment in time
    # β(t) = β₀ × (1 + ε × cos(2π t / 365))
    beta_t = seasonal_beta(t, params)

    # Behavioral dampening factor
    # Φ(t) = 1 / (1 + κ × I_total / N)
    N = POPULATION["N"]
    phi = behavioral_factor(I_total, N, params)

    # Force of infection on each age group
    # λ₁ and λ₂ already incorporate β(t) and Φ(t)
    lambda1, lambda2 = force_of_infection(I1, I2, beta_t, phi, params)

    # -------------------------------------------------------------------------
    # STEP 4: The 10 ODEs
    # -------------------------------------------------------------------------
    # Each line corresponds exactly to one equation from the model spec.
    # Read each line as: "rate of change = inflows - outflows"
    #
    # SIGN CONVENTION:
    #   + means people are ARRIVING into this compartment
    #   - means people are LEAVING this compartment
    #
    # Every flow that leaves one compartment must arrive in another.
    # For example, -lambda1*S1 in dS1 appears as +lambda1*S1 in dE1.
    # This is what guarantees population conservation.

    # --- YOUTH GROUP (group 1) ---

    dS1 = (Lambda1           # births arriving into susceptible pool
         + omega   * R1      # recovered youth losing natural immunity → back to S
         + omega_v * V1      # vaccinated youth losing vaccine protection → back to S
         - lambda1 * S1      # susceptible youth getting infected → to E
         - nu1     * S1      # susceptible youth getting vaccinated → to V
         - mu      * S1)     # natural death

    dE1 = (lambda1 * S1               # infected susceptibles entering latent period
         + (1 - delta) * lambda1 * V1 # breakthrough: vaccinated youth still getting infected
         - sigma * E1                 # exposed youth finishing incubation → to I
         - mu    * E1)                # natural death during latency

    dI1 = (sigma * E1    # exposed youth becoming infectious (finished incubation)
         - gamma * I1    # infectious youth recovering → to R
         - mu    * I1)   # natural death during infectious period

    dR1 = (gamma * I1    # infectious youth recovering
         - omega * R1    # natural immunity waning → back to S
         - mu    * R1)   # natural death

    dV1 = (nu1              * S1   # susceptible youth getting vaccinated
         - (1 - delta)      * lambda1 * V1  # breakthrough infections leaving V
         - omega_v          * V1   # vaccine protection expiring → back to S
         - mu               * V1)  # natural death

    # --- ADULT GROUP (group 2) ---
    # Identical structure to youth, just different parameter values (nu2)
    # and driven by lambda2 instead of lambda1.

    dS2 = (Lambda2
         + omega   * R2
         + omega_v * V2
         - lambda2 * S2
         - nu2     * S2
         - mu      * S2)

    dE2 = (lambda2 * S2
         + (1 - delta) * lambda2 * V2
         - sigma * E2
         - mu    * E2)

    dI2 = (sigma * E2
         - gamma * I2
         - mu    * I2)

    dR2 = (gamma * I2
         - omega * R2
         - mu    * R2)

    dV2 = (nu2              * S2
         - (1 - delta)      * lambda2 * V2
         - omega_v          * V2
         - mu               * V2)

    # -------------------------------------------------------------------------
    # STEP 5: Pack results back into a single array and return
    # -------------------------------------------------------------------------
    # scipy expects a flat array of length 10 in the SAME ORDER as y.
    # y = [S1, E1, I1, R1, V1, S2, E2, I2, R2, V2]
    # So we return derivatives in exactly the same order.

    return np.array([
        dS1, dE1, dI1, dR1, dV1,   # youth group
        dS2, dE2, dI2, dR2, dV2    # adult group
    ])


# =============================================================================
# DIAGNOSTIC HELPERS
# These functions are not used by the solver — they help you understand
# what the model is doing at any given moment, and support validation.
# =============================================================================

def compute_all_intermediates(t: float, y: np.ndarray, params: dict) -> dict:
    """
    Returns all intermediate quantities at a given (t, y) point.

    USE THIS FOR:
    - Debugging: check that beta, phi, lambda are sensible values
    - Validation: confirm behavioral response is working as expected
    - Dashboard: display live force-of-infection alongside compartments

    Parameters
    ----------
    t : float
        Time in days.
    y : np.ndarray, shape (10,)
        Current state vector.
    params : dict
        Model parameters.

    Returns
    -------
    dict with keys:
        'beta_t'    : seasonal transmission rate at time t
        'phi'       : behavioral dampening factor (0 to 1)
        'I_total'   : total infectious count
        'prevalence': I_total / N as a percentage
        'lambda1'   : force of infection on youth
        'lambda2'   : force of infection on adults
    """
    I1 = y[2]
    I2 = y[7]
    I_total = I1 + I2
    N = POPULATION["N"]

    beta_t = seasonal_beta(t, params)
    phi    = behavioral_factor(I_total, N, params)
    lambda1, lambda2 = force_of_infection(I1, I2, beta_t, phi, params)

    return {
        "beta_t"     : beta_t,
        "phi"        : phi,
        "I_total"    : I_total,
        "prevalence" : 100.0 * I_total / N,   # as a percentage
        "lambda1"    : lambda1,
        "lambda2"    : lambda2,
    }


def conservation_residual(y: np.ndarray) -> float:
    """
    Computes how far the current state is from perfect conservation.

    In a correctly implemented model, the sum of all compartments must
    always equal N = 700,000 exactly. Any deviation is a numerical error
    or a bug in the equations.

    Returns
    -------
    float
        |sum(y) - N|
        Should be < 1e-6 at all times during a well-behaved simulation.
        If it grows over time, there is a sign error in the ODEs.
    """
    N = POPULATION["N"]
    return abs(y.sum() - N)


# =============================================================================
# SELF-CHECK (run with: python model/equations.py)
# =============================================================================

if __name__ == "__main__":
    from model.parameters import DEFAULT_PARAMS, INITIAL_STATE

    print("=" * 60)
    print("EQUATIONS.PY — SELF-CHECK")
    print("=" * 60)

    t0 = 0.0
    y0 = INITIAL_STATE
    p  = DEFAULT_PARAMS

    # Add kappa if not already present (safety check)
    if "kappa" not in p:
        p = p.copy()
        p["kappa"] = 5.0

    # ------------------------------------------------------------------
    print("\n[1] Intermediate quantities at t=0")
    intermediates = compute_all_intermediates(t0, y0, p)
    for key, val in intermediates.items():
        print(f"    {key:<15} = {val:.6f}")

    # ------------------------------------------------------------------
    print("\n[2] ODE output at t=0  (dy/dt, people per day)")
    dydt = seirs_v_odes(t0, y0, p)
    labels = ["dS1", "dE1", "dI1", "dR1", "dV1",
              "dS2", "dE2", "dI2", "dR2", "dV2"]
    for label, val in zip(labels, dydt):
        direction = "growing" if val > 0 else "shrinking" if val < 0 else "stable"
        print(f"    {label} = {val:+12.4f}  ({direction})")

    # ------------------------------------------------------------------
    print("\n[3] Conservation check at t=0")
    residual = conservation_residual(y0)
    print(f"    Sum of initial state:   {y0.sum():,.1f}")
    print(f"    Expected N:             {POPULATION['N']:,}")
    print(f"    Residual |sum - N|:     {residual:.2e}")
    status = "PASS" if residual < 1 else "FAIL — check equations!"
    print(f"    Status: {status}")

    # ------------------------------------------------------------------
    print("\n[4] Conservation check after one ODE step")
    # The sum of all derivatives must equal zero (inflows = outflows).
    # If births = deaths exactly, d/dt[S+E+I+R+V] = 0 for each group.
    dydt_sum = dydt.sum()
    print(f"    Sum of all derivatives: {dydt_sum:.6e}")
    status2 = "PASS" if abs(dydt_sum) < 1e-6 else "FAIL — equations not balanced!"
    print(f"    Status: {status2}")

    # ------------------------------------------------------------------
    print("\n[5] Behavioral adaptation check")
    # Test phi at different prevalence levels
    N = POPULATION["N"]
    for frac in [0.0, 0.01, 0.05, 0.10, 0.20]:
        I_test  = frac * N
        phi_val = behavioral_factor(I_test, N, p)
        contact_reduction = (1 - phi_val) * 100
        print(f"    Prevalence {frac*100:4.0f}% → Φ = {phi_val:.4f}  "
              f"({contact_reduction:.1f}% contact reduction)")

    # ------------------------------------------------------------------
    print("\n[6] Seasonal forcing check")
    # β should be at max in winter (t=0, t=365) and min in summer (t=182)
    for day, season in [(0, "winter peak"), (91, "spring"), 
                        (182, "summer trough"), (273, "autumn"),
                        (365, "winter peak again")]:
        b = seasonal_beta(day, p)
        print(f"    t={day:3d} ({season:<18}) β(t) = {b:.5f}")

    print("\n" + "=" * 60)
    print("All checks complete. If all statuses are PASS, equations.py")
    print("is correctly implemented and ready for solver.py.")
    print("=" * 60)

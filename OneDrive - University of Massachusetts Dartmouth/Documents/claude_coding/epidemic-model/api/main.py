"""
main.py
=======
FastAPI REST API for the SEIRS-V epidemic model.

WHAT THIS FILE DOES
-------------------
Exposes the epidemic model as HTTP endpoints so that:
    - The Streamlit dashboard can call it to get simulation results
    - Anyone with the URL can run simulations programmatically
    - The API auto-documents itself at /docs (OpenAPI/Swagger UI)

WHY A SEPARATE API AND NOT JUST CALLING THE MODEL DIRECTLY
-----------------------------------------------------------
Streamlit CAN call model functions directly in Python — and for
local development that is exactly what happens. The FastAPI layer
exists for deployment: Render can host the API as a standalone
service that the dashboard (and any future frontend or external
tool) calls over HTTP. This separation also means the model is
testable independent of any frontend.

ENDPOINTS
---------
    GET  /health              — confirms the API is running
    GET  /default_params      — returns DEFAULT_PARAMS as JSON
    POST /simulate/single     — one deterministic ODE run
    POST /simulate/ensemble   — full prior predictive ensemble run
    GET  /r0                  — R0 for a given parameter set

REQUEST / RESPONSE FORMAT
--------------------------
All POST endpoints accept JSON bodies matching Pydantic models
defined below. All responses are JSON. Pydantic validates incoming
data automatically — bad types or missing fields return a clear
HTTP 422 error before the model ever runs.

RUNNING LOCALLY
---------------
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI
where you can test every endpoint in the browser.

CORS
----
Cross-Origin Resource Sharing is enabled for all origins so the
Streamlit dashboard (which runs on a different port) can call this
API without browser security blocks.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

from model.parameters import (
    DEFAULT_PARAMS,
    FIXED_PARAMS,
    POPULATION,
    INITIAL_STATE,
    TIME_SPAN,
    T_EVAL,
    N_ENSEMBLE_RUNS,
    ENSEMBLE_SEED,
)
from model.solver   import run_single, run_ensemble
from model.analysis import compute_r0, compute_peak_stats, compute_final_size


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="SEIRS-V Epidemic Model API",
    description=(
        "REST API for an age-structured SEIRS-V epidemic model with "
        "seasonal forcing, prevalence-dependent behavioral adaptation, "
        "and prior predictive ensemble uncertainty quantification."
    ),
    version="1.0.0",
)

# Allow all origins so the Streamlit dashboard can call this API
# regardless of which port or domain it is served from.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PYDANTIC MODELS — REQUEST AND RESPONSE SCHEMAS
# =============================================================================
# Pydantic models define the shape of JSON bodies.
# Every field has a type, a default, and a description.
# FastAPI uses these to:
#   (a) validate incoming requests automatically
#   (b) generate the /docs Swagger UI with field descriptions
#   (c) serialize Python objects into JSON responses

class ModelParams(BaseModel):
    """
    All tuneable parameters for a single simulation run.
    Every field defaults to the DEFAULT_PARAMS value so the
    caller can omit any parameter they don't want to change.
    """
    beta0:   float = Field(default=0.035,      ge=0.001, le=0.2,
                           description="Baseline transmission probability per contact per day")
    epsilon: float = Field(default=0.20,       ge=0.0,   le=0.5,
                           description="Seasonality amplitude (0 = no seasonality)")
    kappa:   float = Field(default=5.0,        ge=0.0,   le=50.0,
                           description="Behavioral response strength (0 = no adaptation)")
    sigma:   float = Field(default=1/5,        ge=1/21,  le=1.0,
                           description="E to I progression rate (1/incubation days)")
    gamma:   float = Field(default=1/10,       ge=1/30,  le=0.5,
                           description="I to R recovery rate (1/infectious days)")
    omega:   float = Field(default=1/180,      ge=1/365, le=1/30,
                           description="Natural immunity waning rate (1/immunity days)")
    delta:   float = Field(default=0.80,       ge=0.0,   le=1.0,
                           description="Vaccine efficacy (0=none, 1=perfect)")
    nu1:     float = Field(default=0.002,      ge=0.0,   le=0.02,
                           description="Youth vaccination rate per day")
    nu2:     float = Field(default=0.001,      ge=0.0,   le=0.02,
                           description="Adult vaccination rate per day")

    def to_full_params(self) -> dict:
        """
        Merges user-supplied sampled params with FIXED_PARAMS to produce
        a complete parameter dict ready for the model functions.
        """
        return {
            "beta0":   self.beta0,
            "epsilon": self.epsilon,
            "kappa":   self.kappa,
            "sigma":   self.sigma,
            "gamma":   self.gamma,
            "omega":   self.omega,
            "delta":   self.delta,
            "nu1":     self.nu1,
            "nu2":     self.nu2,
            **FIXED_PARAMS,
        }


class EnsembleRequest(BaseModel):
    """Request body for ensemble runs."""
    n_runs: int = Field(
        default=N_ENSEMBLE_RUNS, ge=10, le=1000,
        description="Number of ensemble members (10 to 1000)"
    )
    seed: int = Field(
        default=ENSEMBLE_SEED,
        description="Random seed for Latin Hypercube Sampling"
    )


class CompartmentSeries(BaseModel):
    """Time series for one compartment."""
    name:   str
    values: list[float]


class SingleSimResponse(BaseModel):
    """Response from a single deterministic simulation run."""
    t:              list[float]
    compartments:   list[CompartmentSeries]
    peak_stats:     dict
    final_size:     dict
    r0:             float
    success:        bool


class EnsembleSimResponse(BaseModel):
    """Response from a full ensemble run."""
    t:              list[float]
    median:         list[CompartmentSeries]
    lower:          list[CompartmentSeries]
    upper:          list[CompartmentSeries]
    r0_distribution: list[float]
    r0_median:      float
    r0_lower:       float
    r0_upper:       float
    peak_stats:     dict
    n_runs:         int
    n_failed:       int


# =============================================================================
# HELPER
# =============================================================================

COMPARTMENT_NAMES = [
    "S1", "E1", "I1", "R1", "V1",
    "S2", "E2", "I2", "R2", "V2"
]

def _array_to_series(arr: np.ndarray) -> list[CompartmentSeries]:
    """
    Converts a (N_timepoints, 10) array into a list of CompartmentSeries.
    Each series has a name and a list of float values.
    JSON does not support numpy floats natively, so we cast to Python float.
    """
    return [
        CompartmentSeries(
            name=COMPARTMENT_NAMES[i],
            values=[float(v) for v in arr[:, i]]
        )
        for i in range(10)
    ]


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", tags=["status"])
def health_check():
    """
    Confirms the API is running and reachable.
    The dashboard calls this on startup to verify connectivity.
    """
    return {
        "status":     "ok",
        "model":      "SEIRS-V age-structured epidemic model",
        "version":    "1.0.0",
        "population": POPULATION["N"],
    }


@app.get("/default_params", tags=["parameters"])
def get_default_params():
    """
    Returns the default parameter set (point estimates / distribution means).
    The dashboard uses this to populate slider initial values.
    """
    # Return only the sampled parameters — fixed params are not user-facing
    sampled = {k: v for k, v in DEFAULT_PARAMS.items()
               if k not in FIXED_PARAMS}
    return {
        "params":     sampled,
        "population": POPULATION,
        "time_span":  {"start": TIME_SPAN[0], "end": TIME_SPAN[1],
                       "days": int(TIME_SPAN[1])},
    }


@app.post("/simulate/single", response_model=SingleSimResponse,
          tags=["simulation"])
def simulate_single(params: ModelParams = None):
    """
    Runs one deterministic ODE simulation with the given parameters.

    If no parameters are provided, uses DEFAULT_PARAMS.

    Returns the full time series for all 10 compartments plus:
        - Peak statistics (peak day, peak infectious count, etc.)
        - Final size (cumulative infection fraction over 5 years)
        - R0 computed via the next-generation matrix

    This endpoint powers the baseline curve in the dashboard and
    the "what-if" scenario mode where the user adjusts a single slider.

    Typical response time: under 1 second.
    """
    if params is None:
        full_params = DEFAULT_PARAMS
    else:
        full_params = params.to_full_params()

    try:
        result = run_single(params=full_params)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"ODE solver error: {str(e)}")

    if not result.success:
        raise HTTPException(status_code=500,
                            detail=f"Solver did not converge: {result.message}")

    peak   = compute_peak_stats(result.y, result.t)
    fsize  = compute_final_size(result.y)
    r0_val = compute_r0(full_params)

    return SingleSimResponse(
        t=[float(v) for v in result.t],
        compartments=_array_to_series(result.y),
        peak_stats={k: round(float(v), 4) for k, v in peak.items()},
        final_size={k: round(float(v), 6) for k, v in fsize.items()},
        r0=round(r0_val, 4),
        success=result.success,
    )


@app.post("/simulate/ensemble", response_model=EnsembleSimResponse,
          tags=["simulation"])
def simulate_ensemble(request: EnsembleRequest = None):
    """
    Runs the full prior predictive ensemble.

    Draws N parameter sets via Latin Hypercube Sampling, runs the ODE
    once per set, and returns the median trajectory plus 95% credible
    interval bands for all 10 compartments.

    Also returns the distribution of R0 values across ensemble members,
    summarized as a list and as median/lower/upper statistics.

    Typical response time: 30-90 seconds for N=500.
    Use n_runs=50 for quick testing.

    The dashboard calls this once on load (or when the user clicks
    "Run Ensemble") and caches the result for the session.
    """
    if request is None:
        n_runs = N_ENSEMBLE_RUNS
        seed   = ENSEMBLE_SEED
    else:
        n_runs = request.n_runs
        seed   = request.seed

    try:
        ens = run_ensemble(n=n_runs, seed=seed, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Ensemble run error: {str(e)}")

    # R0 distribution across ensemble members
    from model.analysis import compute_r0_ensemble
    r0_vals = compute_r0_ensemble(ens.param_list)

    # Peak stats from the median trajectory
    from model.analysis import compute_ensemble_peak_stats
    peak = compute_ensemble_peak_stats(ens)

    return EnsembleSimResponse(
        t=[float(v) for v in ens.t],
        median=_array_to_series(ens.median),
        lower=_array_to_series(ens.lower),
        upper=_array_to_series(ens.upper),
        r0_distribution=[round(float(v), 4) for v in r0_vals],
        r0_median=round(float(np.percentile(r0_vals, 50)),  4),
        r0_lower= round(float(np.percentile(r0_vals,  2.5)),4),
        r0_upper= round(float(np.percentile(r0_vals, 97.5)),4),
        peak_stats={k: round(float(v), 2) for k, v in peak.items()},
        n_runs=n_runs,
        n_failed=ens.n_failed,
    )


@app.get("/r0", tags=["analysis"])
def get_r0(
    beta0:   float = 0.035,
    epsilon: float = 0.20,
    kappa:   float = 5.0,
    sigma:   float = 0.2,
    gamma:   float = 0.1,
    omega:   float = 0.00556,
    delta:   float = 0.80,
    nu1:     float = 0.002,
    nu2:     float = 0.001,
):
    """
    Computes R0 for a given parameter set via the next-generation matrix.

    Accepts parameters as query string values so it can be called
    directly from a browser URL or a simple fetch() call.

    Example:
        GET /r0?beta0=0.04&delta=0.9&nu1=0.003

    Note: kappa does NOT affect R0. It is accepted here for API
    consistency but has no effect on the returned value.
    This is mathematically expected — see analysis.py for the proof.
    """
    params = {
        "beta0": beta0, "epsilon": epsilon, "kappa": kappa,
        "sigma": sigma, "gamma": gamma,   "omega": omega,
        "delta": delta, "nu1":   nu1,     "nu2":   nu2,
        **FIXED_PARAMS,
    }

    try:
        r0_val = compute_r0(params)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"R0 computation error: {str(e)}")

    threshold = "above" if r0_val > 1.0 else "below"

    return {
        "r0":                   round(r0_val, 4),
        "threshold":            threshold,
        "epidemic_establishes": r0_val > 1.0,
        "note_on_kappa":        (
            "kappa does not affect R0. Behavioral adaptation only acts "
            "after the epidemic establishes, not at the disease-free "
            "equilibrium where R0 is evaluated."
        ),
    }

---
title: SEIRS-V Epidemic Model
emoji: 🦠
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
app_file: dashboard/app.py
pinned: false
---

# SEIRS-V Epidemic Model

An age-structured compartmental epidemic model with seasonal forcing,
prevalence-dependent behavioral adaptation, and prior predictive ensemble
uncertainty quantification. Built in Python with an interactive Streamlit
dashboard and FastAPI backend.

---

## Overview

This project implements a novel SEIRS-V (Susceptible–Exposed–Infectious–
Recovered–Susceptible–Vaccinated) model that extends classical compartmental
epidemiology in three directions simultaneously:

- **Age structure**: two population groups (youth and adults) connected by an
  empirical 2×2 contact matrix, so transmission rates differ across age groups
- **Seasonal forcing**: transmission probability varies with a cosine function
  over a 365-day cycle, producing recurrent epidemic waves
- **Behavioral adaptation**: contact rates are reduced as a smooth function of
  current prevalence, reflecting voluntary avoidance behavior during outbreaks

Rather than reporting a single epidemic trajectory, the model runs a
**prior predictive ensemble** of 500 simulations, each with parameters drawn
from calibrated probability distributions via Latin Hypercube Sampling. Output
is presented as median trajectories with 95% credible intervals.

---

## Model Structure

**State vector** (10 compartments):

```
y = [S₁, E₁, I₁, R₁, V₁,  S₂, E₂, I₂, R₂, V₂]
```

Subscript 1 = youth (ages 0–17, N₁ = 200,000).
Subscript 2 = adults (ages 18+, N₂ = 500,000).

**Force of infection** (what makes this model different from basic SEIR):

```
λᵢ(t) = β(t)  ×  Φ(t)  ×  Σⱼ [ cᵢⱼ × Iⱼ(t) / Nⱼ ]
```

Where:
- `β(t) = β₀(1 + ε·cos(2πt/365))` — seasonal transmission
- `Φ(t) = 1/(1 + κ·I_total/N)` — behavioral dampening
- `C = [[c₁₁, c₁₂], [c₂₁, c₂₂]]` — POLYMOD-inspired contact matrix

**R₀** is computed analytically via the next-generation matrix as the dominant
eigenvalue of a 2×2 matrix K. Notably, the behavioral parameter κ does not
affect R₀ — behavioral adaptation only influences epidemic trajectory after
the disease establishes, not the initial growth rate.

---

## Project Structure

```
epidemic-model/
├── model/
│   ├── __init__.py
│   ├── parameters.py     # all constants, distributions, bounds
│   ├── equations.py      # 10 ODEs, seasonal β, behavioral Φ
│   ├── sampler.py        # Latin Hypercube Sampling, validity filter
│   ├── solver.py         # scipy RK45 integration, ensemble runner
│   └── analysis.py       # R₀ via NGM, peak stats, sensitivity
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI REST endpoints
├── dashboard/
│   └── app.py            # Streamlit interactive dashboard
├── tests/
│   └── test_model.py     # automated validation suite (pytest)
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/epidemic-model.git
cd epidemic-model

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run validation checks

```bash
# Verify individual modules
python model/parameters.py
python model/equations.py
python model/sampler.py
python model/solver.py
python model/analysis.py

# Run full automated test suite
pytest tests/ -v
```

All tests should pass before running the dashboard.

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. The dashboard has two tabs:

- **Single run** — adjust parameter sliders and see instant epidemic curves
- **Ensemble** — run 500 LHS-sampled simulations and explore uncertainty bands

### 4. Start the API (optional)

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive API documentation at `http://localhost:8000/docs`.

---

## Dashboard Features

| Feature | Description |
|---|---|
| Epidemic curves | S, E, I, R, V over 5 years with toggleable compartments |
| Age comparison | Youth vs adult infectious trajectories side by side |
| Behavioral adaptation | Φ(t) plotted against prevalence |
| Ensemble ribbons | Median + 95% credible interval for all compartments |
| R₀ histogram | Distribution of R₀ across 500 ensemble members |
| Sensitivity tornado | Spearman correlation of each parameter with R₀ |
| Parameter histograms | Confirms LHS coverage of the full prior range |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API status check |
| GET | `/default_params` | Returns default parameter values |
| POST | `/simulate/single` | One deterministic simulation run |
| POST | `/simulate/ensemble` | Full prior predictive ensemble |
| GET | `/r0` | R₀ for a given parameter set (query string) |

**Example single simulation request:**

```bash
curl -X POST http://localhost:8000/simulate/single \
  -H "Content-Type: application/json" \
  -d '{"beta0": 0.04, "delta": 0.90, "kappa": 8.0}'
```

---

## Validation

Four validation checks confirm the model is mathematically correct:

1. **Conservation** — population N = 700,000 is constant at every timestep
2. **Degenerate case** — with ε=0, κ=0, ν=0, ω=0 and uniform mixing,
   R₀ matches the scalar SEIR formula R₀ = β₀/γ exactly
3. **Threshold behavior** — epidemic dies out when R₀ < 1, grows when R₀ > 1;
   κ does not affect R₀
4. **Directional sensitivity** — higher ν decreases peak, higher κ decreases
   peak size, higher ε increases wave amplitude, higher ω produces recurrent waves

Run the full suite with `pytest tests/ -v`.

---

## Probabilistic Parameterization

Nine parameters are treated as uncertain and sampled from prior distributions:

| Parameter | Distribution | Rationale |
|---|---|---|
| β₀ | Truncated Normal(0.035, 0.005) | Symmetric measurement error |
| ε | Uniform(0.10, 0.30) | Literature range for respiratory pathogens |
| κ | Uniform(2.0, 10.0) | Behavioral parameters are poorly constrained |
| σ | Gamma(shape=25, scale=0.008) | Positive, right-skewed incubation data |
| γ | Gamma(shape=25, scale=0.004) | Positive, right-skewed recovery data |
| ω | Uniform(1/270, 1/90) | Immunity duration 3–9 months |
| δ | Beta(32, 8) | Efficacy bounded in [0,1], mean=0.80 |
| ν₁ | Uniform(0.001, 0.004) | Policy-dependent, wide uncertainty |
| ν₂ | Uniform(0.0005, 0.002) | Policy-dependent, wide uncertainty |

Latin Hypercube Sampling (500 runs, seed=42) ensures efficient coverage.
Post-sampling validity filters enforce individual parameter bounds and a
biological plausibility check on the σ/γ ratio (0.3 ≤ σ/γ ≤ 4.0).

This is a **prior predictive ensemble**, not a stochastic model and not
full Bayesian inference. The ODEs are deterministic — only the inputs are
uncertain. Each of the 500 runs represents one plausible biological reality.

---

## Limitations

- Contact matrix entries are fixed; POLYMOD-derived uncertainty is not propagated
- σ and γ are sampled independently; a Gaussian copula would better capture
  their biological correlation
- Behavioral adaptation is population-averaged; heterogeneous responses
  by age group or risk perception are not modeled
- No disease-induced mortality compartment
- No demographic age transitions between youth and adult groups
- Spatial structure is not modeled; the population is assumed to mix homogeneously

---

## Dependencies

```
numpy, scipy, pandas, fastapi, uvicorn, pydantic, streamlit, plotly
```

Full list with version pins in `requirements.txt`.

---

## Author

Tanmay Sonawane
University of Massachusetts Dartmouth
BS Mathematics (Applied Statistics) / BS Data Science

"""
app.py
======
Streamlit interactive dashboard for the SEIRS-V epidemic model.

WHAT THIS FILE DOES
-------------------
Provides a browser-based interface where any user can:
    - Adjust epidemic parameters using sliders
    - Instantly see how the epidemic curve changes
    - Run the full 500-member prior predictive ensemble
    - Explore R0 as a distribution, not a single number
    - Compare youth vs adult epidemic trajectories
    - See which parameters drive the most uncertainty (tornado chart)
    - Observe behavioral adaptation dampening over time

HOW TO RUN
----------
    streamlit run dashboard/app.py

Opens automatically at http://localhost:8501

ARCHITECTURE NOTE
-----------------
This file calls the model directly in Python (not via the FastAPI).
For local development, direct Python calls are faster and simpler.
When deployed to Streamlit Community Cloud, the model runs on the
same server as the dashboard. The FastAPI (main.py) is for Render
deployment and external access.

STREAMLIT BASICS FOR REFERENCE
-------------------------------
Streamlit re-runs the entire script from top to bottom every time
the user interacts with a widget (moves a slider, clicks a button).
State that needs to persist between reruns (like ensemble results)
is stored in st.session_state — a dict that survives reruns.

st.cache_data  — caches the return value of a function based on
                 its arguments. Used here for ensemble runs so the
                 user doesn't wait 50 seconds every time they move
                 an unrelated slider.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model.parameters import (
    DEFAULT_PARAMS,
    FIXED_PARAMS,
    POPULATION,
    INITIAL_STATE,
    TIME_SPAN,
    T_EVAL,
    N_ENSEMBLE_RUNS,
    ENSEMBLE_SEED,
    SAMPLED_PARAM_NAMES,
)
from model.equations  import compute_all_intermediates
from model.solver     import run_single, run_ensemble
from model.analysis   import (
    compute_r0,
    compute_peak_stats,
    compute_final_size,
    compute_r0_ensemble,
    compute_ensemble_peak_stats,
    compute_sensitivity,
)
from model.sampler    import draw_ensemble, ensemble_to_array


# =============================================================================
# PAGE CONFIG  (must be first Streamlit call in the file)
# =============================================================================

st.set_page_config(
    page_title="SEIRS-V Epidemic Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================
# Minimal overrides — Streamlit's default theme handles most styling.
# We add a few touches: tighter metric cards, a subtle header rule,
# and monospace for parameter value displays.

st.markdown("""
<style>
    /* Header accent line */
    .main-header {
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    /* Tighter metric cards */
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    /* Monospace for parameter values in expanders */
    .param-value {
        font-family: monospace;
        font-size: 0.85rem;
        color: #495057;
    }
    /* Section divider */
    .section-rule {
        border: none;
        border-top: 1px solid #dee2e6;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# COLOUR PALETTE
# Consistent colours across all Plotly charts.
# =============================================================================

COLORS = {
    "I_total":  "#e63946",   # red      — total infectious
    "I1":       "#f4a261",   # orange   — youth infectious
    "I2":       "#457b9d",   # steel blue — adult infectious
    "S":        "#2a9d8f",   # teal     — susceptible
    "R":        "#6a4c93",   # purple   — recovered
    "V":        "#52b788",   # green    — vaccinated
    "E":        "#e9c46a",   # amber    — exposed
    "phi":      "#264653",   # dark     — behavioral factor
    "ci_band":  "rgba(230,57,70,0.15)",   # red ribbon for CI
    "ci_band2": "rgba(70,123,157,0.15)",  # blue ribbon for age groups
}


# =============================================================================
# HELPERS
# =============================================================================

def build_params_from_sidebar(sidebar_vals: dict) -> dict:
    """Merges sidebar slider values with FIXED_PARAMS."""
    return {**sidebar_vals, **FIXED_PARAMS}


def days_to_years(t: np.ndarray) -> np.ndarray:
    """Converts time array from days to years for x-axis display."""
    return t / 365.0


@st.cache_data(show_spinner=False)
def cached_ensemble(n_runs: int, seed: int):
    """
    Runs the ensemble and caches results.

    st.cache_data caches based on (n_runs, seed) — if either changes,
    the ensemble reruns. If they stay the same (e.g. the user moves an
    unrelated slider), the cached result is returned instantly.

    Returns the ensemble result and the param_list separately because
    EnsembleResult is not directly serializable by Streamlit's cache.
    """
    param_list = draw_ensemble(n=n_runs, seed=seed)
    ens        = run_ensemble(n=n_runs, seed=seed, verbose=False)
    r0_vals    = compute_r0_ensemble(param_list)
    sens       = compute_sensitivity(param_list, r0_vals, ens)
    param_arr  = ensemble_to_array(param_list)
    return ens, r0_vals, sens, param_arr


# =============================================================================
# SIDEBAR — PARAMETER CONTROLS
# =============================================================================

def render_sidebar() -> dict:
    """
    Renders all parameter sliders in the sidebar.
    Returns a dict of the current slider values.
    """
    st.sidebar.title("Model Parameters")
    st.sidebar.caption(
        "Adjust parameters to explore how the epidemic responds. "
        "The main plot updates instantly for single runs."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Transmission")

    beta0 = st.sidebar.slider(
        "β₀  Baseline transmission rate",
        min_value=0.01, max_value=0.10,
        value=float(DEFAULT_PARAMS["beta0"]),
        step=0.001, format="%.3f",
        help="Probability of transmission per contact per day, averaged across the year."
    )

    epsilon = st.sidebar.slider(
        "ε  Seasonality amplitude",
        min_value=0.0, max_value=0.5,
        value=float(DEFAULT_PARAMS["epsilon"]),
        step=0.01, format="%.2f",
        help="How much transmission varies with season. 0 = no seasonality."
    )

    kappa = st.sidebar.slider(
        "κ  Behavioral response",
        min_value=0.0, max_value=20.0,
        value=float(DEFAULT_PARAMS["kappa"]),
        step=0.5, format="%.1f",
        help="How strongly people reduce contacts as prevalence rises. 0 = no behavioral change."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Disease Progression")

    incubation_days = st.sidebar.slider(
        "Incubation period (days)",
        min_value=1, max_value=21,
        value=5, step=1,
        help="Average days from exposure to becoming infectious."
    )
    sigma = 1.0 / incubation_days

    infectious_days = st.sidebar.slider(
        "Infectious period (days)",
        min_value=2, max_value=30,
        value=10, step=1,
        help="Average days a person remains contagious."
    )
    gamma = 1.0 / infectious_days

    st.sidebar.markdown("---")
    st.sidebar.subheader("Immunity")

    immunity_months = st.sidebar.slider(
        "Natural immunity duration (months)",
        min_value=1, max_value=24,
        value=6, step=1,
        help="How long recovered immunity lasts before waning."
    )
    omega = 1.0 / (immunity_months * 30.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Vaccination")

    delta = st.sidebar.slider(
        "δ  Vaccine efficacy",
        min_value=0.0, max_value=1.0,
        value=float(DEFAULT_PARAMS["delta"]),
        step=0.01, format="%.2f",
        help="Fraction of infections blocked by the vaccine."
    )

    nu1 = st.sidebar.slider(
        "ν₁  Youth vaccination rate (/day)",
        min_value=0.0, max_value=0.01,
        value=float(DEFAULT_PARAMS["nu1"]),
        step=0.0001, format="%.4f",
        help="Fraction of susceptible youth vaccinated each day."
    )

    nu2 = st.sidebar.slider(
        "ν₂  Adult vaccination rate (/day)",
        min_value=0.0, max_value=0.01,
        value=float(DEFAULT_PARAMS["nu2"]),
        step=0.0001, format="%.4f",
        help="Fraction of susceptible adults vaccinated each day."
    )

    st.sidebar.markdown("---")

    return {
        "beta0":   beta0,
        "epsilon": epsilon,
        "kappa":   kappa,
        "sigma":   sigma,
        "gamma":   gamma,
        "omega":   omega,
        "delta":   delta,
        "nu1":     nu1,
        "nu2":     nu2,
    }


# =============================================================================
# CHART BUILDERS
# Each chart is its own function — keeps render logic separated and
# makes it easy to add or remove charts independently.
# =============================================================================

def chart_epidemic_curves(result, t_years: np.ndarray) -> go.Figure:
    """
    Main epidemic curve: S, E, I, R, V for both groups combined.
    Shows total infectious with the option to toggle compartments.
    """
    y = result.y
    I_total = y[:, 2] + y[:, 7]
    S_total = y[:, 0] + y[:, 5]
    R_total = y[:, 3] + y[:, 8]
    V_total = y[:, 4] + y[:, 9]
    E_total = y[:, 1] + y[:, 6]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_years, y=I_total, name="Infectious (total)",
        line=dict(color=COLORS["I_total"], width=2.5),
        hovertemplate="Day %{customdata:.0f}<br>Infectious: %{y:,.0f}<extra></extra>",
        customdata=result.t,
    ))
    fig.add_trace(go.Scatter(
        x=t_years, y=S_total, name="Susceptible (total)",
        line=dict(color=COLORS["S"], width=1.5, dash="dot"),
        visible="legendonly",
    ))
    fig.add_trace(go.Scatter(
        x=t_years, y=E_total, name="Exposed (total)",
        line=dict(color=COLORS["E"], width=1.5, dash="dot"),
        visible="legendonly",
    ))
    fig.add_trace(go.Scatter(
        x=t_years, y=R_total, name="Recovered (total)",
        line=dict(color=COLORS["R"], width=1.5, dash="dash"),
        visible="legendonly",
    ))
    fig.add_trace(go.Scatter(
        x=t_years, y=V_total, name="Vaccinated (total)",
        line=dict(color=COLORS["V"], width=1.5, dash="dash"),
        visible="legendonly",
    ))

    fig.update_layout(
        title="Epidemic trajectory — all compartments (5-year projection)",
        xaxis_title="Time (years)",
        yaxis_title="Number of people",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


def chart_age_comparison(result, t_years: np.ndarray) -> go.Figure:
    """
    Side-by-side infectious curves for youth vs adults.
    Demonstrates why age structure matters — the two groups peak
    at different times and magnitudes.
    """
    y = result.y
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_years, y=y[:, 2], name="Infectious — youth (I₁)",
        line=dict(color=COLORS["I1"], width=2),
        hovertemplate="Year %{x:.2f}<br>Youth infectious: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t_years, y=y[:, 7], name="Infectious — adults (I₂)",
        line=dict(color=COLORS["I2"], width=2),
        hovertemplate="Year %{x:.2f}<br>Adult infectious: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Age group comparison — infectious by group",
        xaxis_title="Time (years)",
        yaxis_title="Infectious people",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=350,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


def chart_behavioral_adaptation(result, t_years: np.ndarray,
                                 params: dict) -> go.Figure:
    """
    Plots Φ(t) — the behavioral dampening factor — over time.
    Shows how population-level contact reduction tracks with prevalence.
    A Φ of 0.7 means people are having 30% fewer effective contacts
    than normal at that moment.
    """
    N = POPULATION["N"]
    I_total = result.y[:, 2] + result.y[:, 7]
    phi_vals = 1.0 / (1.0 + params["kappa"] * I_total / N)
    prevalence = 100.0 * I_total / N

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=t_years, y=phi_vals,
        name="Φ(t) — contact rate factor",
        line=dict(color=COLORS["phi"], width=2),
        hovertemplate="Year %{x:.2f}<br>Φ = %{y:.3f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=t_years, y=prevalence,
        name="Prevalence (%)",
        line=dict(color=COLORS["I_total"], width=1.5, dash="dot"),
        hovertemplate="Year %{x:.2f}<br>Prevalence: %{y:.2f}%<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        title="Behavioral adaptation Φ(t) vs prevalence",
        xaxis_title="Time (years)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=350,
        margin=dict(l=60, r=60, t=80, b=60),
    )
    fig.update_yaxes(title_text="Φ(t) — dampening factor (0 to 1)",
                     secondary_y=False, range=[0, 1.05])
    fig.update_yaxes(title_text="Prevalence (%)", secondary_y=True)

    return fig


def chart_ensemble_ribbon(ens, t_years: np.ndarray) -> go.Figure:
    """
    Ensemble uncertainty ribbon for total infectious.
    Solid line = median across 500 runs.
    Shaded band = 95% credible interval.
    Width of band = our uncertainty about the epidemic trajectory.
    """
    I_med = ens.median[:, 2] + ens.median[:, 7]
    I_lo  = ens.lower[:, 2]  + ens.lower[:, 7]
    I_hi  = ens.upper[:, 2]  + ens.upper[:, 7]

    fig = go.Figure()

    # Shaded CI ribbon (upper then lower, filling between)
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_years, t_years[::-1]]),
        y=np.concatenate([I_hi, I_lo[::-1]]),
        fill="toself",
        fillcolor=COLORS["ci_band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="95% credible interval",
        hoverinfo="skip",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=t_years, y=I_med,
        name="Median (I₁ + I₂)",
        line=dict(color=COLORS["I_total"], width=2.5),
        hovertemplate="Year %{x:.2f}<br>Median infectious: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Prior predictive ensemble — total infectious with 95% credible interval",
        xaxis_title="Time (years)",
        yaxis_title="Infectious people",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


def chart_ensemble_age_ribbons(ens, t_years: np.ndarray) -> go.Figure:
    """
    Ensemble ribbons separately for youth (I1) and adults (I2).
    Shows that uncertainty in age-group trajectories can differ —
    youth epidemic timing may be more uncertain than adult timing
    or vice versa, depending on which parameters drive variance.
    """
    fig = go.Figure()

    # Youth ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_years, t_years[::-1]]),
        y=np.concatenate([ens.upper[:, 2], ens.lower[:, 2][::-1]]),
        fill="toself",
        fillcolor="rgba(244,162,97,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Youth 95% CI",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t_years, y=ens.median[:, 2],
        name="Youth median (I₁)",
        line=dict(color=COLORS["I1"], width=2),
    ))

    # Adult ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_years, t_years[::-1]]),
        y=np.concatenate([ens.upper[:, 7], ens.lower[:, 7][::-1]]),
        fill="toself",
        fillcolor=COLORS["ci_band2"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Adult 95% CI",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t_years, y=ens.median[:, 7],
        name="Adult median (I₂)",
        line=dict(color=COLORS["I2"], width=2),
    ))

    fig.update_layout(
        title="Ensemble uncertainty by age group",
        xaxis_title="Time (years)",
        yaxis_title="Infectious people",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=380,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


def chart_r0_histogram(r0_vals: np.ndarray) -> go.Figure:
    """
    Histogram of R0 across ensemble members.
    The spread of this distribution represents our uncertainty
    about R0 given uncertain biological parameters.
    A vertical line at R0=1 marks the epidemic threshold.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=r0_vals,
        nbinsx=40,
        name="R0 distribution",
        marker_color=COLORS["I_total"],
        opacity=0.75,
        hovertemplate="R0 = %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    # Threshold line at R0 = 1
    fig.add_vline(
        x=1.0,
        line=dict(color="#264653", width=2, dash="dash"),
        annotation_text="R₀ = 1 threshold",
        annotation_position="top right",
    )

    r0_med = float(np.median(r0_vals))
    fig.add_vline(
        x=r0_med,
        line=dict(color="#e63946", width=1.5),
        annotation_text=f"Median R₀ = {r0_med:.2f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title="R₀ distribution across ensemble members",
        xaxis_title="R₀ (basic reproduction number)",
        yaxis_title="Number of ensemble members",
        height=350,
        margin=dict(l=60, r=20, t=80, b=60),
        bargap=0.05,
    )
    return fig


def chart_sensitivity_tornado(sens: dict) -> go.Figure:
    """
    Tornado chart: Spearman correlation of each parameter with R0.
    Longer bar = stronger influence on R0.
    Red = positive (higher parameter → higher R0).
    Blue = negative (higher parameter → lower R0).
    """
    names = [x[0] for x in sens["r0_sensitivity"]]
    rhos  = [x[1] for x in sens["r0_sensitivity"]]

    # Readable parameter labels
    label_map = {
        "beta0": "β₀ transmission rate",
        "epsilon": "ε seasonality",
        "kappa": "κ behavioral response",
        "sigma": "σ progression rate",
        "gamma": "γ recovery rate",
        "omega": "ω immunity waning",
        "delta": "δ vaccine efficacy",
        "nu1": "ν₁ youth vaccination",
        "nu2": "ν₂ adult vaccination",
    }
    labels = [label_map.get(n, n) for n in names]
    bar_colors = [COLORS["I_total"] if r > 0 else COLORS["I2"] for r in rhos]

    fig = go.Figure(go.Bar(
        x=rhos,
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}<br>Spearman ρ = %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=0, line=dict(color="#495057", width=1))

    fig.update_layout(
        title="Parameter sensitivity — Spearman correlation with R₀",
        xaxis_title="Spearman rank correlation (ρ)",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=180, r=40, t=80, b=60),
    )
    return fig


# =============================================================================
# METRIC CARDS
# =============================================================================

def render_single_metrics(result, params: dict):
    """Displays key summary statistics as metric cards above the main chart."""
    r0     = compute_r0(params)
    peak   = compute_peak_stats(result.y, result.t)
    fsize  = compute_final_size(result.y)
    N      = POPULATION["N"]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="R₀",
        value=f"{r0:.3f}",
        delta="above threshold" if r0 > 1 else "below threshold",
        delta_color="inverse",
        help="Basic reproduction number. >1 means epidemic grows."
    )
    col2.metric(
        label="Peak infectious",
        value=f"{peak['peak_I_total']:,.0f}",
        help=f"Maximum total infectious count ({100*peak['peak_I_total']/N:.1f}% of population)"
    )
    col3.metric(
        label="Peak timing",
        value=f"Year {peak['peak_year']:.2f}",
        help=f"Day {peak['peak_day']:.0f} of the simulation"
    )
    col4.metric(
        label="5-yr infection rate",
        value=f"{100*fsize['final_size_total']:.1f}%",
        help="Fraction of total population infected at least once over 5 years"
    )
    col5.metric(
        label="Φ at peak",
        value=f"{1/(1+params['kappa']*peak['peak_I_total']/N):.3f}",
        help="Behavioral dampening at epidemic peak. Lower = more contact reduction."
    )


def render_ensemble_metrics(ens, r0_vals: np.ndarray):
    """Summary metrics for the ensemble section."""
    peak = compute_ensemble_peak_stats(ens)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Median R₀",
        f"{np.median(r0_vals):.3f}",
        f"95% CI: [{np.percentile(r0_vals,2.5):.2f}, {np.percentile(r0_vals,97.5):.2f}]"
    )
    col2.metric(
        "Median peak infectious",
        f"{peak['peak_I_total_median']:,.0f}",
        f"95% CI: [{peak['peak_I_total_lower']:,.0f}, {peak['peak_I_total_upper']:,.0f}]"
    )
    col3.metric(
        "Median peak year",
        f"{peak['peak_year_median']:.2f}",
        f"95% CI: [{peak['peak_year_lower']:.2f}, {peak['peak_year_upper']:.2f}]"
    )
    col4.metric(
        "Ensemble size",
        f"{ens.trajectories.shape[0]} runs",
        f"{ens.n_failed} failed"
    )


# =============================================================================
# MAIN RENDER
# =============================================================================

def main():
    """Main dashboard layout and rendering logic."""

    # --- Header ---
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("SEIRS-V Epidemic Model")
    st.caption(
        "Age-structured compartmental model with seasonal forcing, "
        "prevalence-dependent behavioral adaptation, and prior predictive "
        "ensemble uncertainty quantification."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Sidebar sliders ---
    sidebar_vals = render_sidebar()
    params       = build_params_from_sidebar(sidebar_vals)
    t_years      = days_to_years(T_EVAL)

    # =========================================================================
    # TAB LAYOUT
    # Two tabs: Single Run (instant response) and Ensemble (50s compute)
    # =========================================================================
    tab_single, tab_ensemble = st.tabs([
        "Single run  (instant)",
        "Ensemble  (prior predictive, ~50s)"
    ])

    # =========================================================================
    # TAB 1 — SINGLE DETERMINISTIC RUN
    # =========================================================================
    with tab_single:

        st.markdown(
            "Adjust the sliders on the left to explore how individual "
            "parameters shape the epidemic. The model solves the 10 ODEs "
            "instantly with your chosen values."
        )

        # Run the ODE with current slider values
        with st.spinner("Solving ODE system..."):
            result = run_single(params=params)

        if not result.success:
            st.error(f"Solver did not converge: {result.message}")
            return

        # --- Key metrics ---
        render_single_metrics(result, params)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # --- Main epidemic curve ---
        st.plotly_chart(
            chart_epidemic_curves(result, t_years),
            use_container_width=True
        )
        st.caption(
            "Toggle compartments on and off by clicking the legend. "
            "S and R are hidden by default — click to show."
        )

        # --- Age comparison and behavioral adaptation side by side ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.plotly_chart(
                chart_age_comparison(result, t_years),
                use_container_width=True
            )
            st.caption(
                "Youth (orange) typically peaks earlier and higher due to "
                "greater within-group contact rates (c₁₁ = 9.0 vs c₂₂ = 4.5)."
            )

        with col_right:
            st.plotly_chart(
                chart_behavioral_adaptation(result, t_years, params),
                use_container_width=True
            )
            st.caption(
                "Φ(t) = 1 when no disease is present. It falls as prevalence "
                "rises, reflecting voluntary contact reduction. κ = 0 disables "
                "this effect entirely."
            )

        # --- Parameter summary expander ---
        with st.expander("Current parameter values"):
            p_col1, p_col2 = st.columns(2)
            sampled_items = [(k, v) for k, v in params.items()
                             if k not in FIXED_PARAMS]
            half = len(sampled_items) // 2
            with p_col1:
                for k, v in sampled_items[:half]:
                    st.markdown(f"`{k}` = **{v:.6f}**")
            with p_col2:
                for k, v in sampled_items[half:]:
                    st.markdown(f"`{k}` = **{v:.6f}**")

    # =========================================================================
    # TAB 2 — ENSEMBLE (PRIOR PREDICTIVE)
    # =========================================================================
    with tab_ensemble:

        st.markdown(
            "The prior predictive ensemble runs the model 500 times, each "
            "with a different parameter set drawn from the prior distributions "
            "via Latin Hypercube Sampling. The shaded bands represent 95% "
            "credible intervals — the range of epidemic trajectories consistent "
            "with our biological uncertainty."
        )

        col_run, col_seed, col_n = st.columns([2, 1, 1])

        with col_n:
            n_runs = st.number_input(
                "Ensemble size", min_value=20, max_value=1000,
                value=N_ENSEMBLE_RUNS, step=50,
                help="Larger = more stable uncertainty bands, slower compute."
            )
        with col_seed:
            seed = st.number_input(
                "Random seed", min_value=0, max_value=9999,
                value=ENSEMBLE_SEED,
                help="Change seed to verify results are robust."
            )
        with col_run:
            run_button = st.button(
                "Run ensemble",
                type="primary",
                help="Runs the full prior predictive ensemble. Takes ~50 seconds."
            )

        # Run or load from cache
        if run_button or "ensemble_done" not in st.session_state:
            with st.spinner(
                f"Running {n_runs} ODE integrations via Latin Hypercube "
                f"Sampling (seed={seed})... This takes about 50 seconds."
            ):
                ens, r0_vals, sens, param_arr = cached_ensemble(n_runs, seed)
            st.session_state["ensemble_done"] = True
            st.session_state["ens"]       = ens
            st.session_state["r0_vals"]   = r0_vals
            st.session_state["sens"]      = sens
            st.session_state["param_arr"] = param_arr
        else:
            ens       = st.session_state["ens"]
            r0_vals   = st.session_state["r0_vals"]
            sens      = st.session_state["sens"]
            param_arr = st.session_state["param_arr"]

        # --- Ensemble metrics ---
        render_ensemble_metrics(ens, r0_vals)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # --- Ensemble ribbon — total infectious ---
        st.plotly_chart(
            chart_ensemble_ribbon(ens, t_years),
            use_container_width=True
        )
        st.caption(
            "The shaded band is the 95% credible interval: 95% of the 500 "
            "ensemble trajectories fall within this region at each timepoint. "
            "Wider bands indicate greater uncertainty in the epidemic trajectory."
        )

        # --- Age ribbons and R0 histogram side by side ---
        col_left2, col_right2 = st.columns(2)

        with col_left2:
            st.plotly_chart(
                chart_ensemble_age_ribbons(ens, t_years),
                use_container_width=True
            )

        with col_right2:
            st.plotly_chart(
                chart_r0_histogram(r0_vals),
                use_container_width=True
            )
            st.caption(
                "All ensemble members with R₀ > 1 produce growing epidemics. "
                "The spread of this histogram directly reflects uncertainty "
                "in β₀, δ, ν₁, ν₂, and contact structure."
            )

        # --- Tornado chart ---
        st.plotly_chart(
            chart_sensitivity_tornado(sens),
            use_container_width=True
        )
        st.caption(
            "Spearman rank correlation between each parameter and R₀ across "
            "the ensemble. Parameters with long bars drive most of the R₀ "
            "uncertainty. Red = positive association, blue = negative. "
            "Note: κ (behavioral response) does not appear here because "
            "it has no effect on R₀ — only on epidemic size after establishment."
        )

        # --- Raw parameter distributions expander ---
        with st.expander("Sampled parameter distributions (ensemble)"):
            st.caption(
                "Histograms of the 500 sampled parameter values. "
                "Confirms LHS explored the full prior range rather than "
                "clustering near the mean."
            )
            n_params = len(SAMPLED_PARAM_NAMES)
            cols     = st.columns(3)
            for j, name in enumerate(SAMPLED_PARAM_NAMES):
                with cols[j % 3]:
                    fig_hist = go.Figure(go.Histogram(
                        x=param_arr[:, j], nbinsx=25,
                        marker_color="#6a4c93", opacity=0.8,
                        name=name,
                    ))
                    fig_hist.update_layout(
                        title=name, height=200,
                        margin=dict(l=20, r=10, t=40, b=30),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "SEIRS-V age-structured epidemic model | "
        "Tanmay Sonawane, UMass Dartmouth | "
        "Prior predictive ensemble via Latin Hypercube Sampling | "
        "scipy RK45 ODE solver"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

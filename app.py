# Thin vs thick bets
# Deploy: Streamlit Cloud, repo nadyyym/thin-thick-bets, file app.py
# Local: pip install -r requirements.txt then streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import first_cross, logistic, simulate_policy as _simulate

THIN = "#cf222e"
THICK = "#1f6feb"
MUTED = "#57606a"
PAPER = "#f6f8fa"


@st.cache_data(show_spinner=False)
def simulate_policy(**kwargs):
    return _simulate(**kwargs)


def line(fig, x, y, name, color, dash=None):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="lines",
            line=dict(color=color, width=2.6, dash=dash or "solid"),
            hovertemplate="%{x:.2f}  %{y:.3f}",
        )
    )


def style(fig, title, xlab, ylab, ymax=None):
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
        hovermode="x unified",
        font=dict(color="#1f2328"),
    )
    if ymax is not None:
        fig.update_yaxes(range=[-0.02 * ymax, ymax])
    return fig


def fmt_t(x):
    return "not inside T" if x is None else f"t = {x:.1f}"


def finite(x, fmt="{:.2f}", empty="inf"):
    return fmt.format(x) if np.isfinite(x) else empty


st.set_page_config(page_title="Thin vs thick bets", page_icon="\u25a3", layout="wide")

st.sidebar.title("Parameters")
st.sidebar.caption(
    "A win starts a stream of W each period after it hits. Streams add. "
    "s, pi and rho are first-class state. Thin bets do not move them."
)

T = st.sidebar.slider("Time horizon T", 4, 60, 24, 1)
W = st.sidebar.number_input("Income per period per win (W)", min_value=0.0, value=1.0, step=0.1)
theta = st.sidebar.slider("Quality bar theta", 0.2, 3.0, 1.0, 0.05)

st.sidebar.markdown("**Effort (time per bet)**")
e_thin = st.sidebar.slider("Thin effort e", 0.1, 3.0, 0.6, 0.05)
e_thick = st.sidebar.slider("Thick effort e", 0.1, 3.0, 1.0, 0.05)
spend_to_clear = st.sidebar.checkbox(
    "Thick spends the minimum that still clears the bar (e* = theta / s)",
    value=True,
)

st.sidebar.markdown("**The three state variables**")
s0 = st.sidebar.slider("Initial skill s0", 0.2, 3.0, 1.0, 0.05)
pi0 = st.sidebar.slider("Initial market belief pi0", 0.02, 0.80, 0.25, 0.01)
rho = st.sidebar.slider("Relatedness of the next bet rho", 0.0, 1.0, 1.0, 0.05)
st.sidebar.caption(
    "s: quality per unit time. A test is valid only if s * e >= theta. "
    "pi: P(this market is good). Updates only after a valid test. "
    "rho: how much of that posterior the next bet inherits. 1 = same segment. 0 = new market."
)

st.sidebar.markdown("**Learning rates**")
skill_gain = st.sidebar.slider("Skill gain after a valid test", 0.0, 0.40, 0.08, 0.01)
prior_strength = st.sidebar.slider("Prior strength (Beta concentration)", 1.0, 16.0, 4.0, 0.5)
p_censor = st.sidebar.slider("Hit rate under the bar (censored)", 0.0, 0.20, 0.05, 0.01)

st.sidebar.markdown("**Isolate the bar**")
freeze = st.sidebar.checkbox(
    "Hold s and pi fixed (post base case: only the bar differs)",
    value=False,
)
n_sims = st.sidebar.select_slider("Simulations", options=[400, 800, 1500, 3000], value=1500)

thin_valid0 = s0 * e_thin + 1e-12 >= theta
thick_e0 = max(0.12, theta / max(s0, 1e-9)) if spend_to_clear else e_thick
thick_valid0 = s0 * thick_e0 + 1e-12 >= theta
e_star0 = theta / max(s0, 1e-9)

st.title("Thin bets do not test the market")
st.markdown(
    "A thin bet sits under the quality bar. Silence then mixes a bad market with a weak test. "
    "A thick bet clears the bar. Skill **s** and market belief **pi** move only after a valid test. "
    "Relatedness **rho** says how much of that belief the next bet keeps. "
    "A win starts a stream of **W** each period after the hit."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("s0 * e thin", f"{s0 * e_thin:.2f}", "below bar" if not thin_valid0 else "clears bar")
c2.metric("s0 * e thick", f"{s0 * thick_e0:.2f}", "below bar" if not thick_valid0 else "clears bar")
c3.metric("e* = theta / s0", f"{e_star0:.2f}")
c4.metric("pi0", f"{pi0:.2f}")
c5.metric("rho", f"{rho:.2f}", "same market" if rho >= 0.99 else ("new market" if 0.01 >= rho else "partial carry"))

if e_thin >= e_thick:
    st.warning("Thin effort is not smaller than thick effort. The labels still apply; the lines may cross.")
if not thick_valid0:
    st.warning("Thick effort is still under the bar at s0. Both policies start censored.")
if freeze:
    st.info("s and pi are held fixed. Only the bar differs. This is the post base case.")

kwargs = dict(
    theta=theta,
    s0=s0,
    pi0=pi0,
    rho=rho,
    skill_gain=skill_gain,
    prior_strength=prior_strength,
    p_censor=p_censor,
    T=float(T),
    W=float(W),
    n_sims=int(n_sims),
    seed=7,
    freeze_s=freeze,
    freeze_pi=freeze,
)

with st.spinner("Running worlds..."):
    thin = simulate_policy(e=e_thin, spend_to_clear=False, **kwargs)
    thick = simulate_policy(e=e_thick, spend_to_clear=spend_to_clear, **kwargs)

t = thin["t"]
t50_thin = first_cross(t, thin["p_any"], 0.5)
t80_thin = first_cross(t, thin["p_any"], 0.8)
t50_thick = first_cross(t, thick["p_any"], 0.5)
t80_thick = first_cross(t, thick["p_any"], 0.8)

q_thin0 = s0 * e_thin
q_thick0 = s0 * thick_e0
q = np.linspace(0.05, max(2.2 * theta, max(q_thin0, q_thick0) * 1.5, 2.2), 240)
info = logistic(q / theta - 1.0)

fig0 = go.Figure()
line(fig0, q, info, "P(the market answers)", MUTED)
fig0.add_vline(x=theta, line_dash="dot", line_color="#8c959f")
fig0.add_vline(x=q_thin0, line_dash="dash", line_color=THIN)
fig0.add_vline(x=q_thick0, line_dash="dash", line_color=THICK)
fig0.add_annotation(x=theta, y=1.05, text="bar", showarrow=False, font=dict(color=MUTED))
fig0.add_annotation(x=q_thin0, y=0.18, text="thin", showarrow=False, font=dict(color=THIN, size=12))
fig0.add_annotation(x=q_thick0, y=0.18, text="thick", showarrow=False, font=dict(color=THICK, size=12))
style(fig0, "Information from one test vs quality  (q = s * e)", "Quality q = s * e", "P(result informs you about the market)", 1.15)
st.plotly_chart(fig0, use_container_width=True)

left, right = st.columns(2)

fig1 = go.Figure()
line(fig1, t, thick["p_any"], "Thick", THICK)
line(fig1, t, thin["p_any"], "Thin", THIN, dash="dash")
style(fig1, "P(at least one win) vs time", "Time", "Probability", 1.05)
left.plotly_chart(fig1, use_container_width=True)

fig_s = go.Figure()
line(fig_s, t, thick["s"], "Thick s", THICK)
line(fig_s, t, thin["s"], "Thin s", THIN, dash="dash")
style(fig_s, "Expected skill s vs time", "Time", "Skill s")
right.plotly_chart(fig_s, use_container_width=True)

fig_err = go.Figure()
line(fig_err, t, thick["pi_err"], "Thick |pi - m|", THICK)
line(fig_err, t, thin["pi_err"], "Thin |pi - m|", THIN, dash="dash")
style(fig_err, "Market-belief error  E[|pi - m|]  (falls only after valid tests)", "Time", "E[|pi - m|]")
left.plotly_chart(fig_err, use_container_width=True)

fig_es = go.Figure()
line(fig_es, t, thick["estar"], "Thick e*", THICK)
line(fig_es, t, thin["estar"], "Thin e*", THIN, dash="dash")
style(fig_es, "Time the next valid test needs  (e* = theta / s)", "Time", "e*")
right.plotly_chart(fig_es, use_container_width=True)

fig_pi = go.Figure()
line(fig_pi, t, thick["pi"], "Thick pi", THICK)
line(fig_pi, t, thin["pi"], "Thin pi", THIN, dash="dash")
fig_pi.add_hline(y=pi0, line_dash="dot", line_color=MUTED)
style(fig_pi, "Expected market belief pi vs time  (martingale: stays near pi0 on average)", "Time", "pi")
left.plotly_chart(fig_pi, use_container_width=True)

fig_c = go.Figure()
line(fig_c, t, thick["conc"], "Thick a+b", THICK)
line(fig_c, t, thin["conc"], "Thin a+b", THIN, dash="dash")
style(fig_c, "Posterior concentration (how sharp pi is). rho below 1 forgets.", "Time", "a + b")
right.plotly_chart(fig_c, use_container_width=True)

fig3 = go.Figure()
line(fig3, t, thick["flow"], "Thick", THICK)
line(fig3, t, thin["flow"], "Thin", THIN, dash="dash")
style(fig3, "Expected income per period", "Time", "Income / period")
left.plotly_chart(fig3, use_container_width=True)

fig4 = go.Figure()
line(fig4, t, thick["cum"], "Thick", THICK)
line(fig4, t, thin["cum"], "Thin", THIN, dash="dash")
style(fig4, "Expected cumulative income", "Time", "Cumulative income")
right.plotly_chart(fig4, use_container_width=True)

st.subheader("Milestones")
rows = [
    ["P >= 0.5", fmt_t(t50_thin), fmt_t(t50_thick)],
    ["P >= 0.8", fmt_t(t80_thin), fmt_t(t80_thick)],
    ["E[time to first win]", finite(thin["e_first_t"]), finite(thick["e_first_t"])],
    ["P(never a win in T)", f"{thin['p_never']:.2f}", f"{thick['p_never']:.2f}"],
    ["E[bets in T]", f"{thin['e_bets']:.1f}", f"{thick['e_bets']:.1f}"],
    ["E[valid tests in T]", f"{thin['e_valid']:.1f}", f"{thick['e_valid']:.1f}"],
    ["s at T", f"{thin['s_end']:.2f}", f"{thick['s_end']:.2f}"],
    ["pi at T (mean)", f"{thin['pi_end']:.2f}", f"{thick['pi_end']:.2f}"],
    ["E[|pi - m|] at T", f"{thin['pi_err_end']:.3f}", f"{thick['pi_err_end']:.3f}"],
    ["posterior a+b at T", f"{thin['conc_end']:.1f}", f"{thick['conc_end']:.1f}"],
    ["E[period income] at T", f"{thin['flow_end']:.2f}", f"{thick['flow_end']:.2f}"],
    ["E[cumulative income] at T", f"{thin['cum_end']:.2f}", f"{thick['cum_end']:.2f}"],
]
st.dataframe(pd.DataFrame(rows, columns=["", "Thin", "Thick"]), hide_index=True, use_container_width=True)

st.markdown(
    """
**How to read this.** Thin bets spend less time and usually stay under the bar, so **s** and **pi** do not move.
You collect zeros that mix execution with market. Thick bets clear the bar, so each result updates **pi**
and raises **s**. Then the next valid test costs less (**e*** falls) and more tests fit in T.

**s** is skill: quality per unit time. It only rises after a valid test.

**pi** is belief that this market is good. E[pi] stays near pi0 (martingale). Learning shows up as
**E[|pi - m|]** falling and as posterior concentration **a+b** rising.

**rho** is the inheritance. At 1, the next bet is the same segment: the posterior stays.
At 0, it is a new market: **pi** snaps back to **pi0**. Skill still carries; market belief does not.

Tick **Hold s and pi fixed** to recover the post base case (only the bar differs).
If thin effort already clears the bar at **s0**, the two policies collapse. The dispute is the location of the bar.
"""
)

with st.expander("Model"):
    st.latex(r"q_t = s_t \\cdot e_t \\qquad \\text{valid iff } q_t \\ge \\theta")
    st.latex(r"e^*_t = \\theta / s_t")
    st.latex(r"p(\\text{win}\\mid \\text{valid}) = m \\qquad p(\\text{win}\\mid \\text{censored}) = p_{\\text{censor}}")
    st.latex(r"m \\sim \\mathrm{Beta}(\\pi_0 \\kappa,\\,(1-\\pi_0)\\kappa) \\qquad \\pi_t = \\mathbb{E}[m\\mid \\text{valid results}]")
    st.latex(r"s_{t+1} = s_t + \\alpha \\cdot \\mathbf{1}\\{\\text{valid}\\}")
    st.latex(r"\\text{next prior} = \\rho \\cdot \\text{posterior} + (1-\\rho)\\cdot \\text{original prior}")
    st.caption(
        "A success at time t_w pays W from t_w to T. Streams add. "
        "Censored tests do not update s or pi. That is the whole claim."
    )

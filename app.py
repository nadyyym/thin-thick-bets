# Thin vs thick bets
# Deploy: Streamlit Cloud, repo nadyyym/thin-thick-bets, file app.py
# Local: pip install -r requirements.txt then streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    ETA,
    S_MAX,
    THETA_MAX,
    THETA_MIN,
    first_cross,
    p_clears,
    simulate_policy as _simulate,
    skill_after,
)

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


st.set_page_config(page_title="Thin vs thick bets", page_icon="▣", layout="wide")

st.sidebar.title("Parameters")
st.sidebar.caption(
    "The bar is redrawn on every bet, even two bets in the same world. "
    "A win starts a stream of W. Streams add. Kill a line when pi falls; "
    "remaining time hunts with rho."
)

T = st.sidebar.slider("Time horizon T", 4, 60, 24, 1)
W = st.sidebar.number_input("Income per period per win (W)", min_value=0.0, value=1.0, step=0.1)

st.sidebar.markdown("**Effort (time per bet)**")
e_thin = st.sidebar.slider("Thin effort e", 0.1, 2.0, 0.6, 0.05)
e_thick = st.sidebar.slider("Thick effort e", 0.1, 2.0, 1.0, 0.05)

st.sidebar.markdown("**Starting competence**")
s0 = st.sidebar.slider("Initial skill s0", 0.2, 3.0, 1.0, 0.05)
st.sidebar.caption(
    "Sets the start of an exponential learning curve "
    "(Heathcote, Brown and Mewhort 2000). You do not pick a learning rate. "
    "Lower s0 means more room to the ceiling, so each valid test moves you more. "
    "A given effort is more likely to clear a random bar when s0 is higher, "
    "but you never know the bar in advance."
)

st.sidebar.markdown("**Market prior (two different knobs)**")
pi0 = st.sidebar.slider("Base rate  (prior mean of m)", 0.02, 0.80, 0.25, 0.01)
prior_strength = st.sidebar.slider("Prior strength kappa  (confidence)", 1.0, 16.0, 4.0, 0.5)
st.sidebar.caption(
    "Base rate = expected hit rate of a *valid* test. Not confidence. "
    "Kappa = how many comparable observations that number is worth. "
    "Same base rate at kappa=1 is a hunch; at kappa=16 you have already seen this class of market."
)

rho = st.sidebar.slider("Relatedness of the next market rho", 0.0, 1.0, 1.0, 0.05)
pi_stop = st.sidebar.slider("Stop if pi falls below", 0.0, 0.40, 0.10, 0.01)
st.sidebar.caption(
    "After a valid test, if posterior mean pi is below this, kill the line. "
    "Remaining time starts a new market. rho=1 same segment: you idle. "
    "rho=0 new market: reset the base rate and hunt. Skill carries. 0 = never stop."
)

p_censor = st.sidebar.slider("Hit rate under the bar (sloppy wins)", 0.0, 0.20, 0.05, 0.01)
freeze = st.sidebar.checkbox("Hold s and pi fixed (no learning, no stopping)", value=False)
n_sims = st.sidebar.select_slider("Simulations", options=[400, 800, 1500, 3000], value=1500)

with st.sidebar.expander("Bar range and learning (literature defaults)"):
    theta_min = st.slider("theta min (slop)", 0.05, 0.5, THETA_MIN, 0.05)
    theta_max = st.slider("theta max (near-perfect)", 0.5, 1.5, THETA_MAX, 0.05)
    st.caption(
        f"Each bet draws a new theta from Unif[{theta_min:.2f}, {theta_max:.2f}]. "
        "Same world, next bet, different bar. Not once per simulation. "
        f"Learning rate eta={ETA:.2f} per valid test, ceiling s_max={S_MAX:.0f}."
    )

p_thin0 = p_clears(s0, e_thin, theta_min, theta_max)
p_thick0 = p_clears(s0, e_thick, theta_min, theta_max)

st.title("Thin bets do not test the market")
st.markdown(
    "The quality bar is **unknown on every test**, drawn independently in "
    f"[{theta_min:.1f}, {theta_max:.1f}]. Two bets in the same world get two bars. "
    "Complete slop almost never clears. Near-perfect work almost always does, "
    "and would take unbounded time to guarantee. "
    "**s0** is starting competence, not a known test length. "
    "**Base rate** is the expected hit rate of a valid test. "
    "**Kappa** is how sure that number is. "
    "A win starts a stream of **W**. If **pi** falls below the stop line, kill and (if rho is low) hunt."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("P(thin clears one test)", f"{p_thin0:.2f}")
c2.metric("P(thick clears one test)", f"{p_thick0:.2f}")
c3.metric("base rate", f"{pi0:.2f}")
c4.metric("kappa", f"{prior_strength:.1f}")
c5.metric("rho", f"{rho:.2f}", "idle after kill" if rho >= 0.99 else ("new hunt" if 0.01 >= rho else "partial carry"))

if e_thin >= e_thick:
    st.warning("Thin effort is not smaller than thick effort. The labels still apply.")
if freeze:
    st.info("s and pi held fixed. No stopping. Only effort vs a fresh random bar differs.")
if theta_min >= theta_max:
    st.error("theta min must be below theta max.")
    st.stop()

kwargs = dict(
    s0=s0,
    pi0=pi0,
    rho=rho,
    prior_strength=prior_strength,
    p_censor=p_censor,
    T=float(T),
    W=float(W),
    n_sims=int(n_sims),
    seed=7,
    freeze_s=freeze,
    freeze_pi=freeze,
    pi_stop=0.0 if freeze else float(pi_stop),
    theta_min=float(theta_min),
    theta_max=float(theta_max),
)

with st.spinner("Running worlds..."):
    thin = simulate_policy(e=e_thin, **kwargs)
    thick = simulate_policy(e=e_thick, **kwargs)

t = thin["t"]
t50_thin = first_cross(t, thin["p_any"], 0.5)
t80_thin = first_cross(t, thin["p_any"], 0.8)
t50_thick = first_cross(t, thick["p_any"], 0.5)
t80_thick = first_cross(t, thick["p_any"], 0.8)

eff = np.linspace(0.05, max(2.0, e_thin, e_thick) * 1.2, 240)
p_eff = np.array([p_clears(s0, x, theta_min, theta_max) for x in eff])
fig0 = go.Figure()
line(fig0, eff, p_eff, "P(this test clears a random bar)", MUTED)
fig0.add_vline(x=e_thin, line_dash="dash", line_color=THIN)
fig0.add_vline(x=e_thick, line_dash="dash", line_color=THICK)
fig0.add_vrect(x0=theta_min, x1=theta_max, fillcolor="#8c959f", opacity=0.08, line_width=0)
fig0.add_annotation(x=e_thin, y=0.18, text="thin", showarrow=False, font=dict(color=THIN, size=12))
fig0.add_annotation(x=e_thick, y=0.18, text="thick", showarrow=False, font=dict(color=THICK, size=12))
style(
    fig0,
    f"P(one test is valid) vs effort, at s0={s0:.2f}. Shaded: possible bars.",
    "Effort e",
    "P(s * e clears a fresh theta)",
    1.15,
)
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
n_show = np.arange(0, 21)
fig_s.add_trace(
    go.Scatter(
        x=n_show,
        y=[skill_after(s0, int(k)) for k in n_show],
        name="curve vs valid-test count (top axis-ish)",
        mode="lines",
        line=dict(color=MUTED, width=1, dash="dot"),
        visible="legendonly",
    )
)
style(fig_s, "Expected skill s vs time  (exponential to ceiling; from s0)", "Time", "Skill s")
right.plotly_chart(fig_s, use_container_width=True)

fig_v = go.Figure()
line(fig_v, t, thick["p_valid"], "Thick P(valid)", THICK)
line(fig_v, t, thin["p_valid"], "Thin P(valid)", THIN, dash="dash")
style(fig_v, "Share of tests that cleared the (fresh) bar", "Time", "P(valid)", 1.05)
left.plotly_chart(fig_v, use_container_width=True)

fig_a = go.Figure()
line(fig_a, t, thick["active"], "Thick still deploying", THICK)
line(fig_a, t, thin["active"], "Thin still deploying", THIN, dash="dash")
style(fig_a, "Share of worlds still deploying  (stop rule + rho)", "Time", "Active", 1.05)
right.plotly_chart(fig_a, use_container_width=True)

fig_err = go.Figure()
line(fig_err, t, thick["pi_err"], "Thick |pi - m|", THICK)
line(fig_err, t, thin["pi_err"], "Thin |pi - m|", THIN, dash="dash")
style(fig_err, "Market-belief error  E[|pi - m|]  (only valid tests update pi)", "Time", "E[|pi - m|]")
left.plotly_chart(fig_err, use_container_width=True)

fig_c = go.Figure()
line(fig_c, t, thick["conc"], "Thick a+b", THICK)
line(fig_c, t, thin["conc"], "Thin a+b", THIN, dash="dash")
style(fig_c, "Posterior concentration. Kill-and-hunt with rho=0 resets this.", "Time", "a + b")
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
    ["Share of worlds that cleared every test", f"{thin['p_world_all_valid']:.3f}", f"{thick['p_world_all_valid']:.3f}"],
    ["Share of worlds that cleared none", f"{thin['p_world_none_valid']:.3f}", f"{thick['p_world_none_valid']:.3f}"],
    ["E[markets entered]", f"{thin['e_markets']:.2f}", f"{thick['e_markets']:.2f}"],
    ["E[kills]", f"{thin['e_killed']:.2f}", f"{thick['e_killed']:.2f}"],
    ["Share still deploying at T", f"{thin['active_end']:.2f}", f"{thick['active_end']:.2f}"],
    ["s at T", f"{thin['s_end']:.2f}", f"{thick['s_end']:.2f}"],
    ["E[|pi - m|] at T", f"{thin['pi_err_end']:.3f}", f"{thick['pi_err_end']:.3f}"],
    ["E[period income] at T", f"{thin['flow_end']:.2f}", f"{thick['flow_end']:.2f}"],
    ["E[cumulative income] at T", f"{thin['cum_end']:.2f}", f"{thick['cum_end']:.2f}"],
]
st.dataframe(pd.DataFrame(rows, columns=["", "Thin", "Thick"]), hide_index=True, use_container_width=True)

st.markdown(
    """
**Bar.** Every bet draws a new theta. Same world, next bet, different bar.
You never plan e* = theta / s. Thick buys a higher chance that *this* attempt
clears. Because the bar is independent, a thin policy that lucks into a few
clears will start learning and later tests get easier. That snowball is a
consequence of a jittering bar, not of a stable one.
Tick hold-s to see the no-learning case (thin valid rate stays at s0 * e).
If the bar were fixed per world, most worlds would be all-clears or none;
the "cleared every test / cleared none" rows would sit near P(one test).

**s0.** Starting competence. Learning is exponential to a ceiling (Heathcote et al. 2000,
*Psychological Review*). Beginners gain more per valid test because the gap is larger.
You do not choose a skill-gain slider.

**Base rate vs kappa.** Base rate (pi0) is *where* you think m sits: the hit rate of a valid test.
Kappa is *how many past comparable tests that guess is worth*. Confidence lives in kappa, not in pi0.

**Stop and rho.** Valid tests update pi. If pi falls below the stop line, kill.
rho=1: same segment, you idle. rho=0: new market, skill keeps, belief and m reset, remaining time hunts.
That is how rho shows up in cash. Thin that never clears never learns enough to stop.
"""
)

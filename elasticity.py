"""Isolation and pairwise sweeps of time-to-a-sure-win.

pi = 1 is a limit. Posterior mean goes to m, not to 1. The operational
reading used here: time until 99% (or 95%) of worlds have at least one
win. That is the original P(at least one) chart hitting 1.
"""

from __future__ import annotations

import math

import numpy as np

from model import p_clears, simulate_hits

BASE = dict(
    e=0.6,
    s0=1.0,
    pi0=0.25,
    rho=1.0,
    prior_strength=4.0,
    p_censor=0.05,
    T=72.0,
    W=1.0,
    n_sims=280,
    seed=11,
    freeze_s=False,
    freeze_pi=False,
    pi_stop=0.10,
    theta_min=0.1,
    theta_max=0.9,
    n_lines=1,
    quality_boost=1.0,
    tau_pi=0.80,
)

# Founder scenario: two skilled people, playbook, competitive niche,
# demand is real, tactic is not.
FOUNDERS = dict(
    s0=2.2,
    pi0=0.32,
    rho=0.40,
    prior_strength=5.0,
    p_censor=0.03,
    T=24.0,
    W=1.0,
    n_sims=360,
    seed=21,
    freeze_s=False,
    freeze_pi=False,
    pi_stop=0.12,
    theta_min=0.35,
    theta_max=0.95,
    tau_pi=0.80,
)

SWEEPS = {
    "e": np.array([0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.20, 1.50]),
    "s0": np.array([0.4, 0.7, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]),
    "pi0": np.array([0.06, 0.12, 0.18, 0.25, 0.35, 0.45, 0.60, 0.75]),
    "prior_strength": np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]),
    "rho": np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    "pi_stop": np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30]),
    "p_censor": np.array([0.0, 0.02, 0.05, 0.10, 0.15, 0.20]),
    "theta_min": np.array([0.05, 0.15, 0.25, 0.35, 0.45]),
    "theta_max": np.array([0.55, 0.70, 0.80, 0.90, 1.05, 1.20]),
}

LABELS = {
    "e": "effort e",
    "s0": "initial skill s0",
    "pi0": "base rate pi0",
    "prior_strength": "prior strength kappa",
    "rho": "relatedness rho",
    "pi_stop": "stop line",
    "p_censor": "hit rate under the bar",
    "theta_min": "theta min (slop)",
    "theta_max": "theta max (near-perfect)",
}

PAIRS = [
    ("e", "s0"),
    ("e", "pi0"),
    ("e", "theta_max"),
    ("s0", "pi0"),
    ("pi0", "prior_strength"),
    ("rho", "pi_stop"),
    ("e", "pi_stop"),
]


def _clean_theta(kw: dict) -> dict:
    out = dict(kw)
    tmin = float(out["theta_min"])
    tmax = float(out["theta_max"])
    if tmin >= tmax:
        out["theta_max"] = tmin + 0.15
    return out


def finite(x, cap=1e8):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v >= cap:
        return None
    return v


def jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = jsonable(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [jsonable(x) if isinstance(x, dict) else (finite(x, cap=1e12) if isinstance(x, (int, float, np.floating)) else x) for x in v]
        elif isinstance(v, (np.floating, float)):
            out[k] = finite(v, cap=1e12)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, np.ndarray):
            out[k] = [finite(x, cap=1e12) for x in v.tolist()]
        else:
            out[k] = v
    return out


def pack(h: dict, extra: dict | None = None) -> dict:
    row = {
        "t50": finite(h["t50"]),
        "t80": finite(h["t80"]),
        "t95": finite(h["t95"]),
        "t99": finite(h["t99"]),
        "t50_win": finite(h.get("t50_win")),
        "t90_win": finite(h.get("t90_win")),
        "t_pi50": finite(h["t_pi50"]),
        "p_win": h["p_win"],
        "p_never": h["p_never"],
        "p_pi": h["p_pi"],
        "cum": h["cum"],
        "e_bets": h["e_bets"],
        "e_valid": h["e_valid"],
        "e_markets": h["e_markets"],
        "valid_rate": h["valid_rate"],
        "s_end": h["s_end"],
        "p_clear": None,
    }
    if extra:
        row.update(extra)
    return row


def run_one(overrides: dict, baseline: dict | None = None) -> dict:
    kw = dict(baseline or BASE)
    kw.update(overrides)
    kw = _clean_theta(kw)
    h = simulate_hits(**kw)
    row = pack(h, {k: kw[k] for k in overrides})
    row["p_clear"] = p_clears(kw["s0"], kw["e"] * kw.get("quality_boost", 1.0), kw["theta_min"], kw["theta_max"])
    return row


def isolation(baseline: dict | None = None, metric: str = "t99") -> dict:
    base = dict(baseline or BASE)
    out = {"baseline": jsonable(base), "metric": metric, "params": {}}
    t0_row = run_one({}, base)
    out["at_baseline"] = t0_row
    for name, xs in SWEEPS.items():
        rows = []
        for x in xs:
            rows.append(run_one({name: float(x)}, base))
        ts50 = [r.get("t50") for r in rows]
        ts_win = [r.get("t50_win") for r in rows]
        pnv = [r.get("p_never") for r in rows]
        out["params"][name] = {
            "label": LABELS[name],
            "x": [float(x) for x in xs],
            "rows": rows,
            "elasticity_t50": local_elasticity([float(x) for x in xs], ts50, float(base[name])),
            "elasticity_t50_win": local_elasticity([float(x) for x in xs], ts_win, float(base[name])),
            "elasticity_p_never": local_elasticity([float(x) for x in xs], pnv, float(base[name])),
            "elasticity": local_elasticity([float(x) for x in xs], ts50, float(base[name])),
        }
    return out


def pairwise(pair: tuple[str, str], baseline: dict | None = None, metric: str = "t99") -> dict:
    base = dict(baseline or BASE)
    a, b = pair
    xs = SWEEPS[a]
    ys = SWEEPS[b]
    # coarser for speed
    if len(xs) > 6:
        xs = xs[::2] if len(xs) > 7 else xs
    if len(ys) > 6:
        ys = ys[::2] if len(ys) > 7 else ys
    z_t50 = []
    z_pnever = []
    z_cum = []
    for x in xs:
        rt50, rnv, rcum = [], [], []
        for y in ys:
            r = run_one({a: float(x), b: float(y)}, base)
            rt50.append(r.get("t50"))
            rnv.append(r.get("p_never"))
            rcum.append(r.get("cum"))
        z_t50.append(rt50)
        z_pnever.append(rnv)
        z_cum.append(rcum)
    return {
        "a": a,
        "b": b,
        "la": LABELS[a],
        "lb": LABELS[b],
        "x": [float(v) for v in xs],
        "y": [float(v) for v in ys],
        "z": z_t50,
        "z_t50": z_t50,
        "z_pnever": z_pnever,
        "z_cum": z_cum,
        "metric": "t50",
    }


def local_elasticity(xs, ts, x0) -> dict:
    xs = np.asarray(xs, dtype=float)
    tfin = np.array([np.nan if v is None else float(v) for v in ts], dtype=float)
    i = int(np.argmin(np.abs(xs - x0)))
    # walk out to nearest finite neighbors
    lo = i - 1
    hi = i + 1
    while lo >= 0 and not np.isfinite(tfin[lo]):
        lo -= 1
    while len(xs) > hi and not np.isfinite(tfin[hi]):
        hi += 1
    t_here = tfin[i] if np.isfinite(tfin[i]) else None
    if 0 > lo or hi >= len(xs):
        return {"at": float(x0), "t": finite(t_here), "elasticity": None, "note": "edge or infinite"}
    if (not np.isfinite(tfin[lo])) or (not np.isfinite(tfin[hi])) or tfin[i] is None or (not np.isfinite(tfin[i])) or tfin[i] == 0 or xs[hi] == xs[lo] or x0 == 0:
        return {"at": float(x0), "t": finite(t_here), "elasticity": None, "note": "undefined"}
    dt = tfin[hi] - tfin[lo]
    dx = xs[hi] - xs[lo]
    el = (dt / tfin[i]) / (dx / x0)
    return {
        "at": float(x0),
        "t": finite(t_here),
        "elasticity": float(el),
        "note": "d log t / d log x",
        "x_lo": float(xs[lo]),
        "x_hi": float(xs[hi]),
        "t_lo": float(tfin[lo]),
        "t_hi": float(tfin[hi]),
    }


def founder_grid(overrides: dict | None = None) -> dict:
    base = dict(FOUNDERS)
    if overrides:
        base.update(overrides)
    e_grid = [0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.20, 1.40]
    stop_grid = [0.0, 0.08, 0.12, 0.18]
    modes = [
        dict(name="split: two parallel tests", n_lines=2, quality_boost=1.0),
        dict(name="pool: both polish one test", n_lines=1, quality_boost=2.0),
    ]
    rows = []
    for mode in modes:
        for e in e_grid:
            for stop in stop_grid:
                r = run_one(
                    {
                        "e": e,
                        "pi_stop": stop,
                        "n_lines": mode["n_lines"],
                        "quality_boost": mode["quality_boost"],
                    },
                    base,
                )
                r["mode"] = mode["name"]
                r["e"] = e
                r["pi_stop"] = stop
                r["n_lines"] = mode["n_lines"]
                r["quality_boost"] = mode["quality_boost"]
                r["e_clear"] = (base["theta_max"] / max(base["s0"] * mode["quality_boost"], 1e-9))
                rows.append(r)
    ranked = sorted(rows, key=lambda r: -(r["cum"] if r["cum"] is not None else -1e18))
    best = ranked[0] if ranked else None
    def pick(mode_name, e, stop):
        for r in rows:
            if r["mode"] == mode_name and 1e-9 >= abs(r["e"] - e) and 1e-9 >= abs(r["pi_stop"] - stop):
                return r
        return None

    stop0 = float(base["pi_stop"])
    slop = pick("split: two parallel tests", 0.25, stop0) or pick("split: two parallel tests", 0.25, 0.12)
    polish = pick("split: two parallel tests", 1.20, stop0) or pick("split: two parallel tests", 1.20, 0.12)
    return {
        "baseline": jsonable(base),
        "rows": rows,
        "best": best,
        "slop": slop,
        "polish": polish,
        "e_cliff": float(base["theta_max"] / max(base["s0"], 1e-9)),
        "p_clear_at_e": [
            {"e": e, "p": p_clears(base["s0"], e, base["theta_min"], base["theta_max"])}
            for e in e_grid
        ],
    }

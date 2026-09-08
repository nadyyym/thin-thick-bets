"""Thin vs thick bets.

The quality bar is unknown on every test: theta ~ Unif(theta_min, theta_max)
independently each bet, including two bets in the same world. There is no
world-level theta. Complete slop almost never clears. Near-perfect work
almost always does, and never needs a known target. Valid iff s * e >= theta.

Skill follows an exponential learning curve after valid tests
(Heathcote, Brown and Mewhort 2000). Initial skill is the starting point;
the remaining gap to a ceiling sets how fast you gain. eta is not a choice.

pi0 is the prior mean of market quality m (base rate of a valid test).
kappa is prior strength (how many comparable observations that mean is worth).

If posterior mean pi falls below pi_stop after at least one valid test, kill
the line. Remaining time starts a new market:
    next prior = rho * posterior + (1-rho) * original prior
    next m     = rho * m         + (1-rho) * fresh m
Skill carries. rho = 1 same segment (idle). rho = 0 new hunt.
The bar is never carried: the next test draws a new theta either way.
"""

from __future__ import annotations

import numpy as np

ETA = 0.15
S_MAX = 6.0
THETA_MIN = 0.1
THETA_MAX = 0.9


def logistic(x: np.ndarray, k: float = 14.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))


def first_cross(t: np.ndarray, y: np.ndarray, level: float):
    hits = np.where(y >= level)[0]
    return float(t[hits[0]]) if len(hits) else None


def p_clears(s: float, e: float, theta_min: float, theta_max: float) -> float:
    """P(s * e >= theta) for theta ~ Unif(theta_min, theta_max) on one test."""
    q = s * e
    span = max(theta_max - theta_min, 1e-12)
    if theta_min >= q:
        return 0.0
    if q >= theta_max:
        return 1.0
    return (q - theta_min) / span


def skill_after(s0: float, n_valid: int, eta: float = ETA, s_max: float = S_MAX) -> float:
    cap = max(s_max, s0)
    return cap - (cap - s0) * np.exp(-eta * n_valid)


def simulate_policy(
    e: float,
    s0: float,
    pi0: float,
    rho: float,
    prior_strength: float,
    p_censor: float,
    T: float,
    W: float,
    n_sims: int,
    seed: int,
    freeze_s: bool,
    freeze_pi: bool,
    pi_stop: float,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
    eta: float = ETA,
    s_max: float = S_MAX,
    n_grid: int = 241,
) -> dict:
    a0 = max(pi0, 1e-6) * prior_strength
    b0 = max(1.0 - pi0, 1e-6) * prior_strength
    grid = np.linspace(0.0, T, n_grid)
    e_used = max(e, 0.12)

    any_hit = np.zeros(n_grid)
    n_wins_g = np.zeros(n_grid)
    s_sum = np.zeros(n_grid)
    pi_sum = np.zeros(n_grid)
    err_sum = np.zeros(n_grid)
    conc_sum = np.zeros(n_grid)
    valid_sum = np.zeros(n_grid)
    active_sum = np.zeros(n_grid)
    first_times: list[float] = []
    n_bets_done = np.zeros(n_sims)
    n_valid = np.zeros(n_sims)
    n_markets = np.zeros(n_sims)
    n_killed = np.zeros(n_sims)
    world_rate = np.zeros(n_sims)

    for i in range(n_sims):
        rng = np.random.default_rng(seed + i)
        m = float(rng.beta(a0, b0))
        a, b = a0, b0
        n_val = 0
        n_val_mkt = 0
        skill = float(s0)
        t = 0.0
        first = None
        win_times: list[float] = []
        bets = 0
        markets = 1
        killed = 0
        active = 1.0

        ev_t = [0.0]
        ev_s = [skill]
        ev_pi = [a / (a + b)]
        ev_err = [abs(a / (a + b) - m)]
        ev_conc = [a + b]
        ev_valid = [0.0]
        ev_active = [1.0]

        while True:
            if t + e_used > T + 1e-12:
                break
            t += e_used
            bets += 1
            # Fresh bar for this bet. Same world does not share a theta.
            theta = float(rng.uniform(theta_min, theta_max))
            valid = skill * e_used + 1e-12 >= theta
            if valid:
                n_val += 1
                n_val_mkt += 1
                hit = m > rng.random()
                if not freeze_pi:
                    if hit:
                        a += 1.0
                    else:
                        b += 1.0
                if not freeze_s:
                    skill = float(skill_after(s0, n_val, eta, s_max))
            else:
                hit = p_censor > rng.random()
            if hit:
                win_times.append(t)
                if first is None:
                    first = t
            pi_now = a / (a + b)
            ev_t.append(t)
            ev_s.append(skill)
            ev_pi.append(pi_now)
            ev_err.append(abs(pi_now - m))
            ev_conc.append(a + b)
            ev_valid.append(1.0 if valid else 0.0)
            ev_active.append(active)

            if (
                (not freeze_pi)
                and pi_stop > 0.0
                and n_val_mkt >= 1
                and pi_stop > pi_now
            ):
                killed += 1
                m_fresh = float(rng.beta(a0, b0))
                m = rho * m + (1.0 - rho) * m_fresh
                a = rho * a + (1.0 - rho) * a0
                b = rho * b + (1.0 - rho) * b0
                pi_now = a / (a + b)
                if pi_stop > pi_now:
                    active = 0.0
                    ev_active[-1] = 0.0
                    break
                n_val_mkt = 0
                markets += 1

        n_bets_done[i] = bets
        n_valid[i] = n_val
        n_markets[i] = markets
        n_killed[i] = killed
        world_rate[i] = (n_val / bets) if bets else 0.0
        if first is not None:
            first_times.append(first)

        ev_t_a = np.asarray(ev_t)
        idx = np.searchsorted(ev_t_a, grid, side="right") - 1
        idx = np.clip(idx, 0, len(ev_s) - 1)
        s_sum += np.take(ev_s, idx)
        pi_sum += np.take(ev_pi, idx)
        err_sum += np.take(ev_err, idx)
        conc_sum += np.take(ev_conc, idx)
        valid_sum += np.take(ev_valid, idx)
        active_sum += np.take(ev_active, idx)
        if first is not None:
            any_hit += (grid >= first).astype(float)
        if win_times:
            wt = np.asarray(win_times)
            n_wins_g += np.searchsorted(wt, grid, side="right").astype(float)

    n = float(n_sims)
    flow = W * n_wins_g / n
    dt = np.diff(grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (flow[1:] + flow[:-1]) * dt)])
    p_any = any_hit / n
    ft = np.asarray(first_times, dtype=float) if first_times else np.array([])
    return {
        "t": grid,
        "p_any": p_any,
        "flow": flow,
        "cum": cum,
        "s": s_sum / n,
        "pi": pi_sum / n,
        "pi_err": err_sum / n,
        "conc": conc_sum / n,
        "p_valid": valid_sum / n,
        "active": active_sum / n,
        "e_bets": float(n_bets_done.mean()),
        "e_valid": float(n_valid.mean()),
        "e_markets": float(n_markets.mean()),
        "e_killed": float(n_killed.mean()),
        "e_first_t": float(ft.mean()) if ft.size else float("nan"),
        "p_never": 1.0 - (ft.size / n),
        "p_end": float(p_any[-1]),
        "flow_end": float(flow[-1]),
        "cum_end": float(cum[-1]),
        "s_end": float((s_sum / n)[-1]),
        "pi_end": float((pi_sum / n)[-1]),
        "pi_err_end": float((err_sum / n)[-1]),
        "conc_end": float((conc_sum / n)[-1]),
        "active_end": float((active_sum / n)[-1]),
        "p_valid_end": float((valid_sum / n)[-1]),
        "valid_rate_mean": float(world_rate.mean()),
        "valid_rate_std": float(world_rate.std()),
        "p_world_all_valid": float((world_rate >= 0.999).mean()),
        "p_world_none_valid": float((0.001 >= world_rate).mean()),
    }

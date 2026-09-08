"""Thin vs thick bets: simulation.

A test is valid iff s * e >= theta. Valid tests update skill s and the
Beta posterior on market quality m. Relatedness rho mixes that posterior
with the original prior before the next bet.
"""

from __future__ import annotations

import numpy as np


def logistic(x: np.ndarray, k: float = 14.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))


def first_cross(t: np.ndarray, y: np.ndarray, level: float):
    hits = np.where(y >= level)[0]
    return float(t[hits[0]]) if len(hits) else None


def simulate_policy(
    e: float,
    theta: float,
    s0: float,
    pi0: float,
    rho: float,
    skill_gain: float,
    prior_strength: float,
    p_censor: float,
    T: float,
    W: float,
    n_sims: int,
    seed: int,
    freeze_s: bool,
    freeze_pi: bool,
    spend_to_clear: bool,
    n_grid: int = 241,
) -> dict:
    a0 = max(pi0, 1e-6) * prior_strength
    b0 = max(1.0 - pi0, 1e-6) * prior_strength
    grid = np.linspace(0.0, T, n_grid)
    s_max = 8.0
    e_min = 0.12

    any_hit = np.zeros(n_grid)
    n_wins_g = np.zeros(n_grid)
    s_sum = np.zeros(n_grid)
    pi_sum = np.zeros(n_grid)
    err_sum = np.zeros(n_grid)
    conc_sum = np.zeros(n_grid)
    estar_sum = np.zeros(n_grid)
    first_times: list[float] = []
    n_bets_done = np.zeros(n_sims)
    n_valid = np.zeros(n_sims)

    for i in range(n_sims):
        rng = np.random.default_rng(seed + i)
        m = float(rng.beta(a0, b0))
        a, b = a0, b0
        skill = float(s0)
        t = 0.0
        valid_count = 0
        first = None
        win_times: list[float] = []
        ev_t = [0.0]
        ev_s = [skill]
        ev_pi = [a / (a + b)]
        ev_err = [abs(a / (a + b) - m)]
        ev_conc = [a + b]
        ev_es = [theta / max(skill, 1e-9)]
        bets = 0

        while True:
            estar = theta / max(skill, 1e-9)
            e_used = max(e_min, estar) if spend_to_clear else e
            if t + e_used > T + 1e-12:
                break
            t += e_used
            bets += 1
            valid = skill * e_used + 1e-12 >= theta
            hit = (m if valid else p_censor) > rng.random()
            if hit:
                win_times.append(t)
                if first is None:
                    first = t
            if valid and not freeze_pi:
                if hit:
                    a += 1.0
                else:
                    b += 1.0
                a = rho * a + (1.0 - rho) * a0
                b = rho * b + (1.0 - rho) * b0
            if valid and not freeze_s:
                skill = min(s_max, skill + skill_gain)
            if valid:
                valid_count += 1
            pi_now = a / (a + b)
            ev_t.append(t)
            ev_s.append(skill)
            ev_pi.append(pi_now)
            ev_err.append(abs(pi_now - m))
            ev_conc.append(a + b)
            ev_es.append(theta / max(skill, 1e-9))

        n_bets_done[i] = bets
        n_valid[i] = valid_count
        if first is not None:
            first_times.append(first)

        ev_t_a = np.asarray(ev_t)
        idx = np.searchsorted(ev_t_a, grid, side="right") - 1
        idx = np.clip(idx, 0, len(ev_s) - 1)
        s_sum += np.take(ev_s, idx)
        pi_sum += np.take(ev_pi, idx)
        err_sum += np.take(ev_err, idx)
        conc_sum += np.take(ev_conc, idx)
        estar_sum += np.take(ev_es, idx)
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
        "estar": estar_sum / n,
        "e_bets": float(n_bets_done.mean()),
        "e_valid": float(n_valid.mean()),
        "e_first_t": float(ft.mean()) if ft.size else float("nan"),
        "p_never": 1.0 - (ft.size / n),
        "p_end": float(p_any[-1]),
        "flow_end": float(flow[-1]),
        "cum_end": float(cum[-1]),
        "s_end": float((s_sum / n)[-1]),
        "pi_end": float((pi_sum / n)[-1]),
        "pi_err_end": float((err_sum / n)[-1]),
        "conc_end": float((conc_sum / n)[-1]),
    }

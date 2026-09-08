# Thin bets vs thick bets

The quality bar is unknown **on every test**, including two tests in the same
world. There is no world-level theta. Complete slop almost never clears.
Near-perfect work almost always does, and would take unbounded time to guarantee.

Belief pi never hits 1. It goes to the true hit rate m. A share of worlds
never win: that is the mixture of markets, the same fact as most startups
dying, not a bug. Time to a win is then two numbers: death rate, and speed
among those who do win.

| Symbol | Meaning | Not |
| --- | --- | --- |
| **s0** | starting competence; sets an exponential learning curve | a known test length |
| **pi0** | base rate: expected hit rate of a *valid* test | confidence |
| **kappa** | prior strength: how many comparable observations that base rate is worth | how good the market is |
| **rho** | relatedness of the *next market* after a kill | per-bet decay |
| **theta** | drawn Unif independently each test | a constant for the world |

Valid iff `s * e` clears that test's theta. Skill grows only after valid tests
(Heathcote, Brown and Mewhort 2000: exponential to a ceiling). If posterior
mean pi falls below the stop line, kill. Remaining time starts a new market:
rho = 1 same segment (idle). rho = 0 new hunt (skill carries, belief and m reset).

Three tabs: thin vs thick, isolation and pairwise elasticities of death and
speed, two skilled founders in a competitive niche (split vs pool, slop vs polish).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

Repo: [github.com/nadyyym/thin-thick-bets](https://github.com/nadyyym/thin-thick-bets)

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. New app.
3. Repository `nadyyym/thin-thick-bets`, branch `main`, main file `app.py`.
4. Deploy.

# Thin bets vs thick bets

Interactive model: a bet below the quality bar does not test the market.

Three state variables sit next to effort and the bar:

| Symbol | Meaning | Moves when |
| --- | --- | --- |
| **s** | skill (quality per unit time) | only after a valid test |
| **π** | belief that this market is good | only after a valid test |
| **ρ** | relatedness of the next bet | slider: 1 = same segment, 0 = new market |

A test is valid iff `s · e ≥ θ`. Thin bets usually stay under that line, so s and π stay put. Thick bets clear it, so the next valid test costs less (`e* = θ / s`) and the posterior on the market actually updates.

A win starts a stream of income W. Streams add.

**E[π] is a martingale** (it stays near π₀ on average). Learning shows up as **E[|π − m|]** falling and as posterior concentration rising. **ρ < 1** forgets: the next bet only partly inherits the posterior.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

Repo: [github.com/nadyyym/thin-thick-bets](https://github.com/nadyyym/thin-thick-bets) (public).

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app**.
3. Repository `nadyyym/thin-thick-bets` → branch `main` → main file `app.py`.
4. Deploy.

No secrets. No extra config.

## Defaults

- T = 24, W = 1, θ = 1
- e_thin = 0.6, e_thick = 1.0
- s₀ = 1 (so e* = 1; thin starts under the bar, thick starts on it)
- π₀ = 0.25
- ρ = 1 (same market)
- skill gain 0.08 per valid test
- Tick **Hold s and π fixed** to recover the post’s base case.

## Model

```
q = s · e
valid iff q ≥ θ
p(win | valid) = m          # true market quality, drawn once per world
p(win | censored) = p_censor
π updates from valid Y only
s rises from valid tests only
next prior = ρ · posterior + (1 − ρ) · original prior
```

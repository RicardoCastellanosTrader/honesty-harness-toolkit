# Extension Guide

How to use this toolkit with your own strategy — either inside the engine's
contract or in an isolated sandbox — and how to run the honesty rite around it.
Every section points at real files in this repository; when in doubt, the code
is the guide.

Language note: some engine internals and the worked example are commented in
Spanish (they are verbatim copies from the original study, plus a sealed
pre-registration that cannot be translated without breaking its hash). The
structure is what matters, and it is language-independent.

---

## 1. The strategy contract

A strategy, to this engine, is four pieces — nothing more:

1. **Precomputed features.** Everything path-dependent (moving averages,
   zones, filters) is computed once, outside the kernel, as plain numpy
   arrays. The kernel never computes an indicator.
   Minimal example: `pipelines/toy/toy_features.py` (an EMA matrix, one row
   per candidate period). Real examples: `precalculate_all_data` in
   `engine/lab_historico_numba_v8_3.py:999` (trend-following), and
   `pipelines/mr/mean_reversion_features.py` (mean-reversion, saved to
   `data_cache/{SYM}_mean_reversion.npz`).
2. **A bit-packed universe with a decoder.** Each configuration is one
   integer; fields live in bit ranges; a `decode` function maps the integer
   to human parameters; a `generate_*_configs()` function enumerates only the
   VALID combinations.
   Toy: 6 bits, 61 valid configs (`toy_kernel.generate_toy_configs`,
   `decode_toy_config`). Real: 26 bits / ~20.9M valid configs
   (`decode_config`, `engine/lab_historico_numba_v8_3.py:1207`) and 17 bits /
   8,192 configs (`decode_mean_reversion_config`,
   `pipelines/mr/mean_reversion_kernel.py:717`).
3. **A Numba kernel that returns a fixed-shape matrix.** Decorated
   `@jit(nopython=True, parallel=True)`, it iterates configurations with
   `prange` and returns `n_configs × 7`:
   `[pnl, trades, wins, cancels, max_dd, gross_profit, gross_loss]`.
   Same seven columns in the toy (`run_toy_simulation_numba`), in TF
   (`run_simulation_numba`, `engine/lab_historico_numba_v8_3.py:1302`) and in
   MR (`run_mean_reversion_numba`, `pipelines/mr/mean_reversion_kernel.py:82`).
   This fixed shape is what lets the walk-forward, the bootstrap and the
   robustness filters work without knowing what your strategy does.
4. **A `run_on_slice` wrapper.** It owns warmup, the accounting window and
   dispatch, so callers never index raw arrays. Toy:
   `pipelines/toy/toy_kernel.py`. Real: `engine/lab_historico_numba_v8_3.py:1940`
   and `pipelines/mr/mean_reversion_kernel.py:767`.

If your idea fits these four pieces, you inherit the engine's machinery. If it
does not fit, do NOT bend the engine — go to section 3 and build a sandbox.

## 2. Adding a pipeline, step by step

The mean-reversion pipeline is the proof that this is possible without
touching the engine: it joined the system as **five new files and zero
modified engine modules** (the only integration point outside the lab was the
production bot, which this toolkit does not ship).

Steps, with the toy as your template and MR as the real-scale case:

1. **Features** — write `xx_features.py`: load the parquet, compute your
   per-bar arrays, save or return them.
   Template: `pipelines/toy/toy_features.py` (40 lines). Real:
   `pipelines/mr/mean_reversion_features.py`.
2. **Kernel** — write `xx_kernel.py`: bit layout, `decode_xx_config`,
   `generate_xx_configs`, the `@jit` kernel returning `n × 7`, and
   `run_on_slice`. Copy the toy's skeleton; keep the seven columns.
   Set costs explicitly (the study used 0.10% round-trip commission,
   `COMMISSION_ROUND_TRIP`, `engine/lab_historico_numba_v8_3.py:423-431`).
3. **Walk-forward** — either start with the toy's regime-free version
   (`pipelines/toy/toy_walk_forward.py`: rolling anchors, opt/fwd = 5000/2000
   bars, the proportion used by the study's multi-anchor experiment) or, at
   real scale, mirror `pipelines/mr/mean_reversion_walk_forward.py`, which
   imports the regime apparatus from `engine/regime_walk_forward.py` and
   injects its results under its own `strategy_type` key.
4. **Robustness** — report a neighborhood index next to any winner: the toy's
   `robustez_is` (fraction of ±1-step parameter neighbors profitable
   in-sample) is the minimal version of the study's plateau filter
   (`engine/extractor_gemas.py:229` and `:333`). Read section 5 before
   trusting it.
5. **Divergences** — if your pipeline deliberately deviates from a reference
   implementation, write it down instead of silently fixing it. Example of
   the discipline: `pipelines/mr/KNOWN_DIVERGENCES.md` (MR's trailing stop is
   always-on, unlike TF's bit-gated one; documented, not corrected, because
   the published evidence was produced with exactly that behavior).
6. **Tests** — at minimum: universe/decode round-trip, determinism
   (bit-for-bit re-run equality), and the `n × 7` contract. See
   `tests/test_toy_smoke.py` for all three in ~100 lines.
7. **Do not touch**: `engine/master.py`, `engine/regime_walk_forward.py`,
   `engine/train_regime_model.py`, `engine/regime_features.py`,
   `engine/data_cache.py`. If you think you need to, you want a sandbox.

## 3. Building a sandbox

Some hypotheses don't fit the kernel: different frequency, different data,
different mechanics. The study's answer was never to generalize the engine —
it was an isolated sandbox with **zero imports from the engine**. If the
sandbox and the engine can't contaminate each other, neither can lie for the
other.

The contract (extracted from the two real cases shipped here):

- Own data loaders, point-in-time (`sandboxes/cs_mom/cs_mom.py:74-146`,
  `sandboxes/mfe/run_edge.py:64-158` — both pull public Binance zips, neither
  reads the engine's cache).
- Own control, frozen in the pre-registration: a point-in-time trivial
  benchmark (cs_mom), symmetric random entries with a fixed seed (mfe,
  `run_edge.py:253`), or drift-0 GBM (`engine/placebo_gen.py` — importing
  just the placebo generator is the one allowed exception, it contains no
  strategy logic).
- Phases with firewalls when the run is expensive: mfe ships a mechanics-only
  phase 0 that is forbidden from concluding anything about edge
  (`sandbox_mfe.py:17-29`), a calibration smoke forbidden from looking at the
  primary metric (`smoke_calibration.py:13`), and a decisive run whose
  look-ahead gate aborts on failure (`run_edge.py:341`).
- The code computes its half of the verdict into `results/*.json`; the human
  half goes into section 10 of the pre-registration (see section 4).

Start from `sandboxes/SANDBOX_TEMPLATE/` (`run_sandbox.py` +
`PREREGISTRO.md` + `.gitignore`). Both real sandboxes ended in a negative
verdict — that is what a working harness looks like from the inside.

## 4. The rite: pre-register → hash → (optional OTS) → run → verdict

The order is the protection. Concretely:

1. **Write sections 0–9** of the pre-registration before computing anything:
   hypothesis with a causal mechanism, one cardinal metric, numeric
   thresholds and a three-zone decision rule, placebo/controls with listed
   seeds, all parameters, anti-leakage plan, and an explicit "what will NOT
   be done" list including your own calibrated expectation. Blank template:
   `docs/PREREGISTRO_TEMPLATE.md` (English template: `docs/PREREGISTRATION_TEMPLATE_EN.md`).
2. **Freeze it.** `python examples/toy/placebo_verdict.py --freeze` hashes
   everything above the `FREEZE-BOUNDARY` marker (sha256) into a sidecar
   file. From then on, any edit above the marker is detected and the run
   refuses to start; changes require a dated amendment plus an explicit
   re-freeze.
3. **Optionally timestamp the seal**: `ots stamp <preregistration>.md` —
   OpenTimestamps commits the hash to Bitcoin, so "it was written before the
   results" becomes independently verifiable. The study sealed its
   pre-registration and verdict documents this way; this repo's worked
   example ships its `.ots`.
4. **Run.** `placebo_verdict.py --run` verifies the seal, then computes the
   noise floor and the real result. It enforces the order: no seal, no run
   (`_require_frozen`).
5. **Write the verdict as section 10** of the same document, below the
   marker (the seal stays valid — verified in this repo). Cardinal metric
   against the frozen threshold first; secondary findings after; diagnostics
   marked non-decisive; the label is asymmetric (a robust "no" closes; a
   "yes" only buys the next, harder test). Worked, executed example:
   `examples/toy/PREREGISTRO_TOY.md` §10.

## 5. Reading the placebo

The numbers below are from this repo's own run (20 GBM worlds, seeds
20260919–20260938, `examples/toy/placebo_verdict_results.json`; see the
figure `examples/toy/ladder.png`).

- **The search-size ladder.** The best config on PURE drift-0 noise averaged
  +27% (best of 10), +83% (best of 61), +138% (best of 1000), with wide
  spread (best-of-61 range across worlds: −490% to +379%). Accounting note:
  toy PnL is additive over notional — a sum of per-trade percentage returns,
  no compounding, no ruin stop — so values below −100% are possible. Nothing
  has any edge in those worlds; the bar rises just because you looked more
  times.
  Whatever your search size is, your result competes against THAT bar, not
  against zero — and every parameter you "just tried" grows N.
- **The zones.** Above the floor's p97.5 → necessary, **not sufficient**
  (zone 1); inside [p2.5, p97.5] → indistinguishable from selection noise;
  below p2.5 → worse than noise. In the worked example BTC's best-of-61
  (+566%) landed in zone 1 exactly as the pre-registration predicted —
  because a drift-0 placebo does not model BTC's 2017–2026 drift, and a
  long/short rule captures drift without merit. Post-hoc context, not part
  of the sealed verdict: buy-and-hold over the same window was +1630% —
  the selected "winner" also lost to the sofa.
- **Transfer is the harder test.** `pipelines/toy/toy_walk_forward.py` asks
  the in-sample winner to perform out-of-sample. On noise: mean in-sample
  +86% → mean forward −19% (9 anchors). On BTC: +74% → +6.5%, with 54% of
  forward windows positive (37 anchors) — a coin flip riding drift.
- **A caution on neighborhood robustness.** The ±1-step neighbor index came
  out ≈0.8 on average both on noise and on BTC in the toy runs. EMA-cross
  winners sit on wide plateaus even in noise, so plateau-ness alone does not
  discriminate here. The study used neighborhood as one filter among several
  (CI bootstrap on forward PF, thresholds, cross-cluster survival) — never
  as the verdict.

## 6. What this tool does not do

- It **reduces mechanical self-deception; it does not guarantee honesty**.
  The other half is the protocol: freezing before looking, one metric, no
  pivot, writing the negative down. The tool leaves a record of when the
  rite was followed; it cannot force you to follow it.
- The toy pipeline is pedagogy, not a production backtest: no stops, no
  slippage, no funding, one asset.
- The real engine's regime machinery (GMM clustering, CI bootstrap, W4
  filters, per-regime specialists) ships in `engine/` and runs, but the
  guided example does not exercise it; expect real study-scale walk-forwards
  to take hours per symbol, not seconds.
- No live trading, no exchange keys, no deployment tooling — the study's
  production bot is not part of this repository.
- No feature requests, and support is best-effort ("as is", MIT): the engine
  is published as it was used, because that is its evidentiary value.
- Origin and calibration of every claim here: the study behind this toolkit
  ended **negative or sub-threshold for all 18 strategy families it examined**
  (Castellanos Macias, R., 2026, *Anatomy of a Null Result*, SSRN,
  https://ssrn.com/abstract=7085378, DOI 10.2139/ssrn.7085378; evidence
  repository `el-trading-no-existe-evidencia`,
  DOI [10.5281/zenodo.21229492](https://doi.org/10.5281/zenodo.21229492));
  the protocol is documented at
  DOI [10.5281/zenodo.21838807](https://doi.org/10.5281/zenodo.21838807).

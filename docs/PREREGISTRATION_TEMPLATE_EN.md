# [EXPERIMENT NAME] — Pre-registration (DRAFT — PENDING FREEZE)

**Date:** YYYY-MM-DD · **Author:** [name] · **Status:** DRAFT → FROZEN on YYYY-MM-DD by [director/author]

> Template extracted from the study's real pre-registrations (MFE, CS-MOM,
> real-edge campaign, B1-B8 series — see DOI 10.5281/zenodo.21229492).
> The ten sections below are **mandatory, in this order**; the verdict is
> ALWAYS the last one and is written at closure INTO THIS SAME DOCUMENT
> (ex-ante → result traceability). Tag each section with its inline status:
> `[FROZEN]`, `[PENDING FREEZE]`, `[DONE]`. Spanish original:
> `PREREGISTRO_TEMPLATE.md`.

## 0. Discipline

- This document is FROZEN **before** seeing any result. After the freeze,
  only dated, justified AMENDMENTS are allowed (appended, never rewritten).
- One experiment → one question.
- The backtest is an **ASYMMETRIC FALSIFIER**: a robust "no" closes the
  hypothesis; a "yes" does NOT confirm edge — at most it buys paper trading.
- Isolated sandbox, production untouched, declared budget.

## 1. Causal hypothesis [FREEZE]

One hypothesis, with an explicit causal mechanism (WHY would this edge exist
and who pays for it). If you cannot write the mechanism, you do not have a
hypothesis: you have a pattern.

## 2. Primary-source verifications [DO BEFORE THE FREEZE]

What data actually exists (paths, coverage, gaps), verified by reading it —
not from memory or blogs. List each verification with its evidence.

## 3. Cardinal metric, a priori [FREEZE]

ONE metric decides (e.g. net expectancy per trade with a 95% CI; net Sharpe).
No later pivot: if, upon seeing results, you prefer another metric, that is a
dated amendment BEFORE recomputing — or it does not count.

## 4. Thresholds and falsifier [FREEZE]

Exact numbers: which result kills the hypothesis, which result would earn
paper trading, and which zone remains INCONCLUSIVE. The confidence interval
goes on the number that decides.

## 5. Placebos / controls / multiple testing [FREEZE]

What noise the result is compared against (drift-0 GBM via
`engine/placebo_gen.py`, symmetric random entries, a trivial point-in-time
benchmark…) and how the number of attempts is controlled (Bonferroni/FDR if
m>1 tests; declare m here).

## 6. Analysis plan and a-priori parameters [FREEZE]

Every parameter fixed BEFORE running (periods, windows, costs — realistic
taker commission, slippage if applicable —, universe, seeds). Whatever is not
written here cannot be tuned afterwards.

## 7. Anti-leakage / point-in-time [FREEZE]

How you ensure no decision sees the future: prefix-invariance gate,
backward `merge_asof` for asynchronous data, untouched holdout,
`engine/leakage_gate.py` where applicable. Declare the holdout budget (which
data stays virgin and for what).

## 8. Compute and cost plan [FREEZE]

Disk/CPU/GPU/hours estimated (with a ×5–10 safety factor over your
projection) and monetary budget (ideally $0). If there is a prior calibration
smoke, state what it may look at and what it is FORBIDDEN from looking at.

## 9. What will NOT be done

Explicit anti-cherry-picking list: no unlisted variants will be tried, the
period will not be extended if the result disappoints, the metric will not
change, a negative will not be reframed as "almost". Include your calibrated
expectation of the outcome.

## 10. VERDICT — [written at closure, against the criteria above]

Closure structure (the study's verdict contract):
1. Run metadata + confirmation that the ex-ante gates passed.
2. **The cardinal metric against the frozen threshold, first.**
3. Secondary findings and safeguards (monotonicity, consistency, control).
4. Non-decisive diagnostics, labeled as such.
5. **ASYMMETRIC VERDICT** in bold (POSITIVE→paper trading / NEGATIVE /
   INCONCLUSIVE).
6. The project decision is deferred to a separate conversation — never taken
   inside the record.

---
*Optional sections used by the study when applicable: calibration smoke with
its result, dated amendments, holdout ledger, three-zone stopping criteria.*

# TOY — Selección best-of-N vs suelo de ruido — Pre-registro (CONGELADO 2026-09-19)

**Fecha:** 2026-09-19 · **Autor:** Claude Code (aprendiz), dirección Ricardo · **Estado:** CONGELADO 2026-09-19 (sha256 en sidecar `PREREGISTRO_TOY.md.sha256`, escrito por `placebo_verdict.py --freeze` ANTES de correr)

> Ejemplo RELLENADO de la plantilla del Arnés (`docs/PREREGISTRO_TEMPLATE.md`).
> Propósito didáctico: enseñar el rito completo (prerregistro → hash → run →
> veredicto) sobre la tubería toy. El experimento es deliberadamente ingenuo
> — y el prerregistro lo declara (§9).

## 0. Disciplina [CONGELADA]

- Este documento se congela antes de ver NINGÚN resultado (ni el suelo GBM ni BTC). `placebo_verdict.py` impone el orden: sin sidecar de hash no hay run.
- Un experimento → una pregunta (§1).
- Falsador asimétrico: un "supera el suelo" aquí NO confirma edge (§4, §9); un "no supera" sí es informativo (la selección no distingue BTC de ruido).
- Sandbox: solo tubería toy + placebo canónico del estudio; producción intacta; $0; CPU.

## 1. Hipótesis causal [CONGELADA]

La selección ingenua "mejor configuración de un universo sobre toda la serie" (el recitador) produce sobre BTC un PnL que refleja mayoritariamente (a) la deriva del activo 2017-2026 y (b) el sesgo de selección best-of-N — no una ventaja de la regla EMA-cross. Mecanismo: maximizar sobre N intentos eleva el resultado esperado del ganador aunque la regla no aporte nada (estadística de extremos), y una regla stop-and-reverse long/short captura deriva direccional sin mérito predictivo.

## 2. Verificaciones primary-source [HECHAS 2026-09-19]

- Datos reales: `data_cache/BTCUSDT_1h.parquet`, 79.558 barras 1h, 2017-08-17 04:00 → 2026-09-19 09:00 UTC, descargado de Binance vía `engine/download_full_history.py` (Sesión 2). Sin huecos relevantes verificados por la descarga paginada.
- Generador de placebo: `engine/placebo_gen.py` — `gbm()` drift-0 y `calib()` (σ horaria, rango intrabar, p0, volumen) — funciones canónicas del estudio, intactas (sanitización de la Sesión 2 documentada en el propio archivo).
- Kernel: `pipelines/toy/toy_kernel.py` — `run_ema_cross_pairs_numba` comparte el núcleo `_simulate_rows` con el kernel bit-packed de la Sesión 2 (equivalencia verificada por test `tests/test_toy_smoke.py`).

## 3. Métrica cardinal a priori [CONGELADA]

**PnL neto % de la MEJOR configuración de un universo de 61 (best-of-61) sobre la serie completa**, con: comisión 0,10% round-trip por trade (la del motor), accounting desde la barra 350 (warmup unificado del grid denso), preset laxo (histéresis h=0), cierre contable en la última barra. Una sola métrica; sin pivote.

## 4. Umbrales y criterio de decisión [CONGELADOS]

Sea `S = [p2.5, p97.5]` de la distribución de best-of-61 sobre los 20 mundos GBM drift-0 (§5). Tres zonas sobre `real_61` (best-of-61 en BTC):

1. `real_61 > p97.5(S)` → **SUPERA EL SUELO DRIFT-0**. NO se interpreta como edge (confundido con deriva, §9). En un experimento serio solo habilitaría la fase siguiente (walk-forward + placebo con deriva modelada).
2. `real_61 ∈ S` → **INDISTINGUIBLE DEL RUIDO DE SELECCIÓN** (negativo).
3. `real_61 < p2.5(S)` → **PEOR QUE RUIDO**.

Métrica secundaria (didáctica, NO decide): la escalera del suelo — se espera media(best-of-10) < media(best-of-61) < media(best-of-1000) sobre ruido, retratando que el listón sube con el tamaño de la búsqueda.

## 5. Placebo / controles / multiple-testing [CONGELADOS]

- 20 mundos GBM drift-0 (`placebo_gen.gbm`), longitud = la de la serie BTC (79.558 barras), σ y p0 calibrados sobre BTC real con `placebo_gen.calib` (procedimiento E1 del estudio).
- **Semillas fijas listadas**: 20260919, 20260920, …, 20260938 (BASE_SEED=20260919 + i, i=0..19; `np.random.default_rng`).
- El tamaño de búsqueda REAL declarado es 61 (el universo canónico de la toy); la comparación decisiva es solo real_61 vs suelo_61 (m=1 test; la escalera 10/1000 es descriptiva y no genera decisiones).

## 6. Plan de análisis y parámetros a priori [CONGELADOS]

- Universos: **U61** = canónico toy (8 rápidas {5,8,12,16,21,26,34,42} × 8 lentas {30,40,55,75,100,140,190,250}, fast<slow → 61). **U10** = índices 0,6,12,18,24,30,36,42,48,54 del array canónico ordenado por config_id. **U1000** = primeras 1000 parejas lexicográficas (fast,slow) del grid fast∈{3,5,…,49} × slow∈{20,27,…,335}, fast<slow (el generador produce exactamente 1000; verificado con `--universe` antes del freeze).
- Warmup/accounting: barra 350 en TODOS los runs (real y placebo, los tres tamaños) — comparabilidad exacta.
- Comisión 0,10% RT; histéresis 0; long/short stop-and-reverse; cierre contable al final.
- Estadísticos del suelo por tamaño: media, mediana, p2.5, p97.5, min, max sobre los 20 mundos.
- Software: `examples/toy/placebo_verdict.py` de este repo, entorno Python 3.12 del toolkit.

## 7. Anti-leakage / point-in-time [CONGELADO]

Las EMAs son causales (EMA[t] usa solo barras ≤ t); no hay selección de período post-hoc (la serie BTC es "todo lo disponible hasta hoy", descargada en la Sesión 2 ANTES de diseñar este experimento); los universos y semillas quedan fijados aquí antes de computar nada. No hay holdout: este experimento no lo necesita porque su pregunta es sobre el sesgo de selección full-sample, no sobre generalización (esa es la pregunta del walk-forward, `toy_walk_forward.py`).

## 8. Plan de cómputo y coste [CONGELADO]

CPU local, sin GPU, $0. 21 series × ~80k barras × ≤1071 configs (numba paralelo): estimación < 5 min con factor de seguridad ×5 sobre lo observado en la Sesión 2 (kernel 0,23 s / 61 configs / 80k barras). Sin smoke previo sobre BTC: el gate lo impide y el coste no lo justifica.

## 9. Qué NO se hará [CONGELADO]

- NO se proclamará edge si `real_61` supera el suelo: el placebo drift-0 NO modela la deriva de BTC (~×100 en el período) y una regla long/short la captura sin mérito. **Expectativa calibrada pre-registrada: se ESPERA zona 1** — el valor didáctico es mostrar que "batir al suelo ingenuo" es un listón trivial, y que el listón honesto exige walk-forward (ver `toy_walk_forward.py`) y placebo con deriva.
- NO se cambiará la métrica, ni el universo, ni las semillas, ni el warmup tras ver resultados (cambio = enmienda fechada + re-freeze explícito, que el gate detecta).
- NO se probarán más tamaños de búsqueda que los tres declarados.
- NO se usará este resultado para decidir operar: la toy no tiene stops, ni slippage, ni funding — es pedagogía, no un backtest de producción.

Sello de tiempo opcional (documentado, no requerido): `ots stamp examples/toy/PREREGISTRO_TOY.md.sha256` (OpenTimestamps, atestación Bitcoin diferida; `ots upgrade` horas después), el mismo mecanismo usado para sellar los prerregistros del estudio.

<!-- FREEZE-BOUNDARY: todo lo anterior está congelado; el sha256 del sidecar cubre desde el inicio del archivo hasta esta línea inclusive. El §10 (veredicto) se escribe DEBAJO tras el run. -->

## 10. VEREDICTO (2026-09-19) — ZONA 1: SUPERA EL SUELO DRIFT-0 — NO ES EDGE (esperado, §9)

**Metadatos del run**: `placebo_verdict.py --run`, Python 3.12.14 (venv del toolkit), gate de freeze verificado (sha256 `0519812ac6820616…` coincidente con el sidecar), 20 mundos GBM + BTC real, suelo 2,5 s + real 0,1 s. Calibración sobre BTC: σ_1h=0,00766, p0=81.326, 79.558 barras. Resultados completos: `placebo_verdict_results.json`.

**1. Métrica cardinal contra el umbral congelado (lo primero)**: `real_61 = +565,8%` vs suelo_61 `[p2.5, p97.5] = [−356,6%, +344,8%]` → **ZONA 1** del criterio §4. Conforme a §4 y §9: **NO se interpreta como edge** — el placebo drift-0 no modela la deriva de BTC (~×100 en el período) y la regla long/short la captura sin mérito predictivo. En un experimento serio esto solo habilitaría la fase siguiente (walk-forward + placebo con deriva modelada). La expectativa calibrada pre-registrada (§9: "se ESPERA zona 1") se cumplió.

**2. Secundario — la escalera del suelo (didáctica, no decide)**: media del mejor-de-N sobre ruido puro: N=10 → **+26,8%**; N=61 → **+82,6%**; N=1000 → **+137,9%** (medianas +47,2 / +111,2 / +174,5; máximos +269 / +379 / +402). Monotonía confirmada en media y mediana: **el listón que hay que batir sube solo con mirar más configuraciones**, sin que ninguna tenga nada. El mejor mundo de ruido (seed 20260937) alcanzó +378,9% con 61 configs — dos tercios del resultado "real" de BTC, sin BTC.

**3. Salvaguardas**: mismas semillas listadas en §5, mismo warmup 350, misma comisión, mismos universos congelados en §6 (|U1000|=1000 verificado pre-freeze). El §10 se escribió debajo del marcador de freeze: el hash sellado no cambió.

**4. Diagnóstico no-decisivo**: real_10=+479,4% < real_61=+565,8% < real_1000=+599,9% — en BTC el ganador también crece con la búsqueda, señal clásica de que parte del resultado ES búsqueda.

**VEREDICTO ASIMÉTRICO**: el experimento NO demuestra edge (zona 1 = confundido con deriva, por diseño y por expectativa pre-registrada). Lo que sí queda demostrado didácticamente: (a) el suelo de ruido de best-of-N es enorme y crece con N; (b) el rito prerregistro→hash→run→veredicto es ejecutable en minutos con esta tubería. El siguiente listón honesto para cualquier hipótesis del lector: `toy_walk_forward.py` (caída IS→FWD) y un placebo con deriva modelada.

**Decisión de proyecto**: no aplica (ejemplo pedagógico). Para hipótesis propias: conversación aparte, nunca dentro del acta.

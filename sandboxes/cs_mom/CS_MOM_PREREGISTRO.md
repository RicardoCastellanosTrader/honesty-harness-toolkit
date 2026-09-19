# CS-MOM (Cross-Sectional Momentum, market-neutral) — Pre-registro (BORRADOR v0, PENDIENTE FREEZE)

**Estado:** BORRADOR para revisión + freeze de Ricardo. **NO ejecutar** hasta congelar las
decisiones `[PENDIENTE FREEZE]`. Sandbox aislado `cs_mom_sandbox/`, producción intacta, $0.
Listón recalibrado: "digno por coste-tiempo" (no "transformador").

## 0. Disciplina
Pre-registro congelado antes de ver resultados. Parámetros fijados A PRIORI por lógica, NO
tuneados. Robustez se corre DESPUÉS, debe SOBREVIVIR, no selecciona. Backtest con costes
modelados ≠ fills reales → un "sí" da derecho a paper-trading, NO confirma edge.

## 1. Hipótesis causal [CONGELAR]
El capital en cripto rota entre altcoins (euforia → fuertes siguen subiendo; pánico → débiles
colapsan más). Activos con mayor momentum ajustado por volatilidad en las últimas N → superan
a los de menor momentum en las próximas K. El edge es la **DISPERSIÓN relativa (fuertes −
débiles)**, NO la dirección absoluta del mercado. Respaldo académico (Jegadeesh-Titman y
posterior), cross-sectional (no time-series), neutral al mercado.

## 2. Verificaciones primary-source [HECHAS 2026-06-29]
- **klines 12h disponibles** futures/um/monthly (también 4h/8h/1d). Baratas (no tick).
- **Cobertura de los 45 (CARDINAL — point-in-time / supervivencia):**
  - 34/45 cubren 2023-2024 completo.
  - **11 listados después de 2023-01**: STX 2023-02, ARB 2023-03, SUI 2023-05, WLD 2023-07,
    SEI 2023-08, ONDO 2024-01, ENA 2024-04, TAO 2024-04, RENDER 2024-07, POL 2024-09 (+SHIB).
  - **Rebrands historia partida**: POL←MATIC (2024-09), RENDER←RNDR (2024-07).
  - **Mapeo de símbolo**: SHIB en futures = `1000SHIBUSDT` (no SHIBUSDT). Verificar otros 1000x.
  - Solo 28/45 llegan a 2022-01 (relevante si se incluye el bear 2022).

## 3. Diseño base [parámetros A PRIORI, congelar UNO]
- **Factor**: retorno de últimos N días / ATR (momentum ajustado por volatilidad). Prop **N=7d**.
- **Frecuencia evaluación**: `[PENDIENTE FREEZE]` prop **12h** (menos churn).
- **Ranking** cross-sectional de los símbolos VIVOS a fecha t (point-in-time, §5 Blindaje 4).
- **Carteras**: LONG quintil superior / SHORT quintil inferior. `[PENDIENTE FREEZE]` quintil
  (9+9) vs decil. Prop **quintil** (45 nombres → decil = 4-5, frágil/squeeze).
- **Ponderación**: volatility-targeting ATR-inverse (fórmula existente) → vol(LONG)=vol(SHORT).
- **Holding/rebalanceo**: `[PENDIENTE FREEZE]` prop **2-3d con histéresis** (margen a priori §5).

## 4. Blindajes del documento original [AFIRMADOS]
- **B1 — lado corto PESIMISTA**: funding del corto (pagado cada 8h, descontado del PnL del
  corto), short-squeeze SIN suavizar (subidas violentas tal cual ocurrieron, cola adversa real
  vía bar HIGH). Si el edge solo vive ignorando esto, es espejismo.
- **B2 — costes de rebalanceo COMPLETOS**: 0.10% taker por lado en CADA cambio (abrir/cerrar/
  ajustar, no solo entradas), TODAS las transacciones. Slippage por iliquidez (§5 Blindaje 6).
  Histéresis reduce churn — margen congelado por lógica, NO tuneado.
- **B3 — sin cherry-picking universo/período**: 45 símbolos, todas las rotaciones, período
  continuo que incluye euforia y pánico.

## 5. Blindajes AÑADIDOS en la revisión (cardinales)
- **Blindaje 4 — POINT-IN-TIME + SUPERVIVENCIA (el más grave):**
  - El ranking a fecha t usa SOLO símbolos ya listados a t (los 11 tardíos entran en su fecha;
    universo crece ~34→45). NO rankear un símbolo antes de su listado.
  - Rebrands: decisión a priori — stitch MATIC→POL y RNDR→RENDER (continuidad de momentum) o
    tratar como listados nuevos. Prop: **stitch** (la historia económica es continua).
  - **Sesgo de supervivencia irreductible con este universo**: los 45 son los que SOBREVIVIERON
    a 2026; las altcoins que murieron/delistaron en 2023-24 (las que el corto habría ganado o el
    largo perdido) están AUSENTES. Dirección del sesgo ambigua pero REAL → el backtest es "sobre
    universo de supervivientes", caveat permanente, paper-trading imprescindible aun con "sí".
- **Blindaje 5 — funding de AMBAS patas**: en cripto el LONG de alto-momentum en euforia PAGA
  funding positivo alto (no solo el corto). Netear funding de las dos patas, no solo el corto.
- **Blindaje 6 — slippage por iliquidez a priori**: proxy objetivo congelado (p.ej. ADV del
  dataset `metrics`, o volumen medio de kline), `slip_bps = f(1/ADV)` con tope; NO tuneado.

## 6. Métrica y contraste [recalibrado, con reframe]
- **Métrica cardinal**: SHARPE + MAX DRAWDOWN + duración del peor DD (no retorno total).
- **Neutralidad verificada (cardinal)**: BETA de L−S vs mercado ≈ 0. Si beta>0, la "ganancia"
  puede ser exposición alcista en un mercado que subió (2024 bull) → no es el factor, es beta.
- **Contraste — REFRAME (importante):** una estrategia market-neutral con Sharpe positivo y
  beta≈0 es valiosa por su DESCORRELACIÓN, aunque su Sharpe sea menor que el buy-and-hold de un
  bull. Por tanto:
  - GATE primario = **L−S Sharpe positivo y robusto + beta≈0 + retorno L−S > 0 tras corto
    pesimista + costes completos + sobrevive ambos regímenes**.
  - Buy-and-hold (índice EW de los 45 / BTC) = **CONTEXTO**, no el gate (en sample bull-pesado
    exigir batir su Sharpe mataría un edge neutral genuino). Se reporta, no decide solo.
- **Robustez (después, no selección)**: N∈{5,7,14}, quintil vs decil, holding {1,2,3}. El edge
  no puede depender de un punto paramétrico.
- **Look-ahead gate**: factor usa solo datos ≤ t (cierre de barra de decisión), entrada a t+1;
  prefix-invariance como en MFE/Exp#2.

## 7. Período [PENDIENTE FREEZE]
- Prop **2022-01 → 2024-12 point-in-time** (28→45 símbolos): captura BEAR real (2022) + chop
  (2023) + bull (2024) → satisface "sobrevive ambos regímenes" con datos, mejor que 2023-24
  (casi todo alcista). Alternativa: 2023-2024 (34→45) si se prefiere universo más completo.

## 8. Veredicto
- **DIGNO** (primer "sí" posible): L−S Sharpe positivo robusto, beta≈0, retorno positivo tras
  corto pesimista + costes, robusto a parámetros, sobrevive bull Y bear → candidato a
  paper-trading (NO confirmación; survivorship + fills reales pendientes en vivo).
- **NO DIGNO**: ganancia de beta oculta, o muere bajo corto/costes, o solo un régimen, o no
  robusto → 7ª línea (la de mayor respaldo previo), "no" definitivo.
- **Expectativa calibrada**: CS-MOM da retornos modestos, ruidosos, con drawdowns largos
  (crashes de factor históricos). Lo probable: edge modesto, real, descorrelacionado, comido en
  parte por corto+costes → "digno por coste-tiempo" es el listón, no algo transformador.

## 9. DECISIONES PENDIENTES DE FREEZE (recomendación)
1. **Frecuencia**: 4h vs 12h → prop **12h**.
2. **Holding**: 1/2/3d → prop **2d con histéresis**.
3. **Quintil (9+9) vs decil** → prop **quintil** (45 nombres, decil frágil/squeeze).
4. **Benchmark**: EW-45 vs BTC → prop **EW-45**, pero como CONTEXTO (ver reframe §6).
5. **Margen histéresis**: por lógica, p.ej. entra en top-20%, sale solo al caer de top-33%
   (banda ~½ quintil); simétrico en corto. Congelado, no tuneado.
6. **Símbolos historia incompleta**: point-in-time (entran en su fecha) + decisión rebrand
   (prop stitch MATIC→POL, RNDR→RENDER) + mapeo 1000SHIB.
7. **[NUEVO] Período**: 2022-2024 point-in-time (bear real) vs 2023-2024.
8. **[NUEVO] Reframe del gate**: ¿aceptas L−S Sharpe+beta≈0+costes como gate y buy-and-hold
   como contexto (recomendado para neutral), o exiges batir el Sharpe del buy-and-hold?

---

## 10. VEREDICTO (2026-06-29) — NO DIGNO (7ª línea, la de mayor respaldo previo)
Benchmark EW point-in-time Sharpe 1.443 (2023 2.01 / 2024 1.04). Universo 35→45 point-in-time,
stitch POL=MATIC+POL, RENDER=RNDR+RENDER, SHIB=1000SHIB. Costes solo-en-rotación (corregido),
funding bilateral, slippage por iliquidez, beta vs EW y BTC.

| Curva | Sharpe full | 2023 | 2024 | maxDD | veredicto |
|---|---|---|---|---|---|
| A Long-Only | 1.163 | 1.081 | 1.244 | −53.9% | NO DIGNA (<1.94; **ni bate el EW 1.44**) |
| B Short-Only | −0.832 | −1.139 | −0.659 | −85.6% | VENENO (esperanza negativa confirmada) |
| C L/S neutral | 0.541 | **0.028** | 0.929 | −27.4% | NO DIGNA (Sharpe<1; DD fuera de −15/−20) |

**Curva C — neutralidad REAL lograda**: beta_EW=−0.043, beta_BTC=−0.041 (∈[−0.15,0.15]). Es la
ÚNICA línea del proyecto con retorno positivo, real y descorrelacionado (Sharpe 0.54, +29% 2a),
PERO sub-umbral: falla pilar rentabilidad (0.54<1.0) y protección (−27%>target), y en 2023 fue
plano (0.028 — todo el PnL vino de 2024). El momentum cross-sectional cripto EXISTE y es neutral,
pero demasiado modesto para el listón coste-tiempo.

**Curva A**: el momentum long-only (Sharpe 1.16) NO bate ni el comprar-y-mantener EW (1.44) → el
tilt direccional no aporta sobre lo trivial. **Curva B**: short = veneno (funding+squeezes),
refuta "la basura colapsa = ganancia fácil". **Turnover 1.657/rebal** (~83%/pata) → momentum poco
persistente, corrobora el Sharpe modesto.

**Cláusula 2024**: A y C sobreviven 2024 (Sharpe>0) pero no alcanzan el listón → la supervivencia
de régimen no rescata un base sub-umbral. **Robustez OMITIDA** (no se rescata un base muerto).

**VEREDICTO ASIMÉTRICO: NO DIGNO.** A no bate el benchmark; C es neutral-real pero sub-umbral
(2 de 3 pilares fallan); el corto es veneno. Sesgo de supervivencia = techo optimista → el C real
es aún menor. 7ª línea convergente, la de mayor respaldo académico previo. La señal cross-sectional
es real y neutral (lo "menos muerto" del proyecto) pero no "digna por coste-tiempo". Decisión
proyecto/bot = conversación Ricardo aparte.

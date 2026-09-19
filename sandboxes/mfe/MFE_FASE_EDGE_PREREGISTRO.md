# MFE — Fase de EDGE — Pre-registro (CONGELADO 2026-06-29)

**Estado:** **CONGELADO** por Ricardo 2026-06-29 (7 decisiones + 2 precisiones + test
secundario). Tras este freeze: smoke de calibración (autorizado) → si los buckets prometen
N suficiente y el coste es viable → run de 2 años. La Fase 0 (mecánica) ya está validada
(`sandbox_mfe.py` + `results/`). Sandbox aislado, producción intacta, $0 capital.

**Hallazgo cardinal (primary-source) que condiciona el diseño:** las cascadas bajistas
ocurren cuando el funding CAE, no en euforia. El bucket >5e-4 (Escenario A) es ~1.2-1.4%
de las ventanas 8h Y **casi toda la euforia de 2 años se concentra en 2024-03** (15 de 26
ventanas >5e-4 BTC / 15 de 30 ETH). Tensión interna: el momento que predeciría la mejor
reversión (funding alto + cascada bajista) puede ser raro POR MECÁNICA (las dos condiciones
tienden a excluirse). Si >5e-4 sale mudo NO es fracaso sino hallazgo; la señal del funding
(si existe) viviría en ≤0 vs 0–5e-4 → de ahí el **test secundario** (§5).

---

## 0. Disciplina (heredada del proyecto)
Pre-registro congelado **antes** de ver resultados. Un experimento → una pregunta.
El backtest es **FALSADOR ASIMÉTRICO**: puede dar un "no" robusto, NO un "sí" confiable
(señal incompleta sin ping de liquidación + sin libro L2 → slippage real no simulable).
Un "sí" solo da derecho a **paper-trading en vivo**, no es confirmación de edge.

## 1. Hipótesis causal [CONGELADA]
Liquidation Exhaustion: tras una cascada bajista (liquidación de largos), la presión
vendedora se agota y el precio rebota. **Extensión funding (covariable de régimen):**
funding positivo extremo = momentum alcista persistente (demanda dispuesta a pagar el
apalancamiento). Cruzado con cascada bajista:
- **Escenario A** (funding alto, cascada CONTRA momentum): demanda estructural intacta →
  reversión más violenta → mayor supervivencia.
- **Escenario B** (funding ≤0): sin demanda subyacente → el precio sigue sangrando.

El funding es **"viento de cola"**. Se prueba por estratificación, **no se asume por filtro**.

## 2. Verificaciones primary-source [HECHAS 2026-06-29]
- **aggTrades** futures/um/daily: exhaustivos pero agregados (price/qty/ts ms /
  is_buyer_maker). Suficientes para velocidad + volumen direccional.
- **forceOrder / liquidaciones históricas: NO EXISTEN** en data.binance.vision. → En
  backtest la firma = velocidad + volumen. El ping solo entra EN VIVO (paper-trading).
- **Sin libro L2 reconstruible** (solo bookDepth periódico) → slippage real exige vivo.
- **fundingRate** futures/um/**monthly** (no daily): `calc_time(ms), funding_interval_hours,
  last_funding_rate`. Cadencia **8h** (respetar `funding_interval_hours` por fila).
  Cobertura **2020-01 → 2026-05** (BTC y ETH). Anotación as-of: `funding_at_signal` =
  `last_funding_rate` del último `calc_time ≤ t_cascada` (sin look-ahead).
- **Distribución funding 2023-2024** (covariable, mirarla no contamina el outcome):
  - BTC: ≤0 = 9.3% (203) / 0–5e-4 = 89.6% (1964) / >5e-4 = **1.2% (26)**.
  - ETH: ≤0 = 6.7% (146) / 0–5e-4 = 92.0% (2017) / >5e-4 = **1.4% (30)**.
  - **CONCERN:** el bucket euforia (>5e-4) es escaso y las cascadas anti-correlacionan
    con funding alto → Escenario A puede salir casi vacío. Ver §6 (regla N-mínimo + smoke).

## 3. Las 6 decisiones congeladas del backtest base [CONGELADAS]
1. **Muestra continua sin cherry-picking**: período largo, TODAS las cascadas de la firma
   objetiva (las que revirtieron y las que no). Más eventos → alargar período, nunca
   seleccionar los violentos.
2. **Slippage pesimista (falsador duro)**: entrada forzada al PEOR precio de los siguientes
   3s de aggTrades (para FADE long = el precio más ALTO en esos 3s) + 0.20% RT fees.
3. **Salida por tiempo 1 / 3 / 5 min + hard stop −2% MAE** (si toca −2% intra-ventana, sale
   en stop). **Métrica cardinal = tasa de supervivencia** (% cascadas con retorno neto > 0
   bajo slippage), reportando N + distribución + peor MAE. NO PnL/PF agregado (outliers).
4. **Contraste simétrico estratificado**: random-walk en momentos aleatorios del mismo
   período, **maquinaria COMPLETA idéntica** (entrada por exhaustión, stop −2%, salida por
   tiempo, slippage pesimista) + estratificado por funding en los mismos buckets.
5. **Horizontes** 1 / 3 / 5 min.
6. **Veredicto asimétrico**: muere bajo slippage → "no" robusto, cerrado. Sobrevive →
   derecho a paper-trading en vivo (NO confirmación; señal incompleta sin ping + slippage
   real no simulable).

## 4. Extensión funding (anotación pasiva + estratificación) [CONGELADA en forma]
- Se añade SOLO la columna `funding_at_signal` a CADA cascada Y a CADA entrada de control.
- La firma detecta las mismas cascadas, las opera igual. Cero cambio en §3.
- Análisis post-hoc: supervivencia por bucket de funding, para estrategia Y control.
- **Métrica cardinal de la estratificación = BRECHA dentro de bucket**:
  `gap(b) = surv_cascada(b) − surv_control(b)`. Aísla funding×cascada del confound
  funding×mercado-general. La cascada sigue siendo la única variable, dentro de cada nivel.

## 5. Salvaguardas multiple-testing + tests [CONGELADAS]
1. **Buckets + dirección monótona pre-registrados**: predicción = `gap(≤0) ≤ gap(0–5e-4) ≤
   gap(>5e-4)`. Si NO es monótono → hipótesis FALLA, no se rescata con otra historia.
2. **Consistencia cross-símbolo y cross-horizonte**: el patrón debe aparecer en BTC Y ETH
   y en los 3 horizontes (magnitud decayendo con el tiempo). Una celda aislada = ruido.
3. **Control simétrico TAMBIÉN estratificado** (cardinal): la pregunta es si la BRECHA crece
   con el funding, no si "cascadas con funding alto > cascadas con funding bajo".
4. **Test PRIMARIO**: CI95 bootstrap (10k) de `gap(>5e-4) − gap(≤0)` excluye 0 **y**
   monotonía across 3 buckets **y** consistencia BTC+ETH. **DEPENDE de N(>5e-4) ≥ 30.**
5. **Test SECUNDARIO** (independiente del bucket raro, por el hallazgo cardinal): monotonía
   sobre los buckets con N suficiente **y** `gap(0–5e-4) − gap(≤0)` con CI95 que excluye 0.
   Operativo si >5e-4 es fantasma (N<30) → el experimento dice algo sobre estratificación en
   el rango POBLADO en vez de quedar mudo.
6. **Robustez (NO selección)**: tras el primario, matriz {0.3%,0.5%}×{z≥3,z≥4}. El resultado
   debe SOBREVIVIR la matriz para ser creíble; no se elige la celda ganadora.

## 6. Decisiones (7) [CONGELADAS por Ricardo 2026-06-29]
1. **Período + símbolos**: 2023-01-01 → 2024-12-31 continuo, BTC+ETH.
2. **Entrada / "exhaustión"**: opción (ii) — entrada en el primer segundo tras el disparo en
   que el retorno trailing-15s deja de ser negativo (descompresión), espera máx 60s; si no,
   entrada forzada a +60s. Sin look-ahead. Respaldada por el −4.96% MAE de Fase 0.
3. **Umbrales firma (primario)**: caída ≥0.3%/15s & vol z≥3.0 (uno solo, evita
   threshold-hunting). Matriz de robustez {0.3,0.5}×{3,4} DESPUÉS, no como selección (§5.6).
4. **Dirección**: solo cascadas BAJISTAS (liquidación de largos). Squeeze de cortos = aparte.
5. **Cortes bucket**: fijos ≤0 / 0–5e-4 / >5e-4 + **regla N-mínimo 30** (debajo → se reporta
   pero NO se interpreta). **Precisión**: el smoke reporta el CONTEO por bucket como su PRIMER
   output, antes de cualquier tasa de supervivencia (la decisión primario/secundario se toma
   sobre el conteo, no contaminada por una tasa seductora).
6. **Control**: 10 entradas aleatorias por cascada (≥5.000/símbolo), semilla fija, hereda la
   maquinaria COMPLETA idéntica + estratificación en los mismos buckets.
7. **Test del gap**: primario y secundario congelados en §5.4–5.5. Slippage de salida: salida
   al último precio del segundo-horizonte (0.20% RT ya cubre fees); slippage pesimista solo
   en ENTRADA + hard stop −2%.

## 7. Plan de cómputo / disco [PENDIENTE FREEZE]
- **Disco (Caveat #13 mandatorio):** procesar **día a día** — descargar 1 día → detectar
  cascadas + entradas control + slippage (raw aggTrades) + anotar funding → **append**
  resultados → **borrar el día crudo** → siguiente. Nunca retener toda la data
  (2 años × 2 sym ≈ 70–115 GB si se guardara; con cleanup el pico es <1–2 GB).
- **Smoke de calibración OBLIGATORIO antes del run completo** (disciplina compute Frame 3):
  1 mes continuo (p.ej. 2024-08) × BTC+ETH, **reportando solo mecánica**: seg/día,
  GB/día, nº cascadas, **conteo cascadas-por-bucket** (para validar viabilidad de buckets).
  **Sin mirar supervivencia en el smoke** (Blindaje 1 recursivo).
- **Estimación run completo** (recalibrar con smoke + factor 5–10×): ~730 días/sym; el
  cuello es descarga + parseo raw 3s-slippage. ETA y disco se fijan tras el smoke.

## 8. Veredicto de la estratificación [CONGELADO]
- BRECHA crece monótonamente con funding, consistente BTC+ETH y 3 horizontes →
  funding = filtro de régimen estructural genuino → refuerza derecho a paper-trading CON
  condición de funding. NO confirmación de edge.
- Brecha no-monótona / celda aislada / control estratificado muestra el mismo patrón
  (= era el mercado) → estratificación no aporta → decisión sobre la tasa global.
- En AMBOS casos el backtest sigue siendo falsador; un positivo da derecho a paper-trading,
  no a desplegar capital.

## 9. Nota de continuidad
El "v2.6 funding filter" fue REFUTADO empíricamente (contrarian, memoria
`project_v2_6_status`). Esto **no es** re-correr aquello: es funding como **covariable de
estratificación sobre microestructura** (no filtro), con control simétrico estratificado.
Metodológicamente más limpio y en dominio nuevo.

---

## 10. SMOKE DE CALIBRACION — RESULTADO (2026-06-29, mes 2024-03, mecanica + conteo)
**Conteo cascadas por bucket (output primario, SIN supervivencia):**

| bucket | BTC (mar) | ETH (mar) | proyeccion 2 anios (hi via ratio ventanas) |
|---|---|---|---|
| neg (<=0) | 0 | 0 | sin cobertura en smoke (mar = mes alcista; se puebla en meses bajistas del run) |
| mid (0-5e-4) | 65 | 102 | miles (sobrado >=30) |
| hi (>5e-4) | 19 | 14 | ~33 BTC / ~28 ETH |

- **Escenario A NO es fantasma**: 19 BTC / 14 ETH cascadas bajistas reales con funding
  5.5e-4..10e-4 (caidas ~-0.34% en 15s durante el run ATH de marzo). Proyeccion ~28-33 a
  2 anios -> **BTC cruza N>=30, ETH borderline (~28)**. Proyeccion posiblemente OPTIMISTA
  (marzo = mes pico de volatilidad euforica) -> rango honesto hi ~25-35.
- **neg sin cobertura en el smoke** (marzo fue puro alcista, 0 ventanas <=0). El run lo
  puebla desde meses bajistas (203 BTC / 146 ETH ventanas <=0 en 2 anios, donde las cascadas
  se concentran). Alta confianza de que sera el bucket mas poblado, pero el smoke NO lo midio.
- **Decision primario vs secundario**: NO se pre-elige; la regla N>=30 ya congelada
  auto-selecciona por simbolo sobre los conteos REALES del run de 2 anios.
- **Coste/disco VIABLE** (mejor de lo temido): ~37 GB transfer total (no 70-115), pico disco
  <1-2 GB con cleanup dia-a-dia, ~1.9h wall solo-deteccion (2anios x 2sym). Smoke: 5 min.
- **wait exhaustion mediano = 15s** (entrada (ii) funciona; un window tras el disparo).

## 11. Refinamientos recomendados ANTES del run (pendiente OK Ricardo, NO cambian hipotesis/zonas)
- **Cluster-bootstrap por ventana-8h** para los CI del gap: las cascadas dentro de una
  ventana 8h comparten funding Y estan temporalmente agrupadas -> el bootstrap a nivel evento
  sobrestima precision. Resamplear a nivel ventana-8h (o dia) hace el CI honesto. Coherente
  con el caveat N-efectivo ya congelado; es rigor estadistico, no cambio de hipotesis.
- **Arquitectura del run**: una pasada con warmup de baseline 1h (carry del grid del dia
  previo) + buffer look-forward de 5min para horizontes/slippage; raw borrado tras cada dia
  (~37 GB single-pass). Deteccion sobre serie continua (sin artefacto de frontera diaria).

---

## 12. ADDENDUM (fase posterior — NO afecta el run actual ni su freeze) — 2026-06-29
**Hipotesis de seleccion de universo (Ricardo):** si la MFE es viable, desplazar la seleccion
de simbolos de "top por capitalizacion" hacia los MAS VOLATILES (excluyendo rug pulls).
Causal (Ruta 2): la dislocacion tras una cascada depende de cuanta liquidez forzada se vuelca
vs profundidad de libro -> mas volatil = dislocaciones mayores = mas sobre-extension = mas
espacio para que la reversion supere el slippage. El edge (si existe) seria mayor donde la
fisica es mas intensa. NO es "probar otros simbolos a ver".

**Diagnostico ya disponible en el run actual (NO veredicto):** contraste BTC-vs-ETH de la
BRECHA cascada-vs-control (ETH tipicamente mas volatil que BTC).
- gap(ETH) > gap(BTC) -> senal temprana de que el eje volatilidad importa (respalda fase
  posterior por volatilidad).
- gap(BTC) ~ gap(ETH) pese a dif. de volatilidad -> evidencia EN CONTRA de la intuicion.
- **CAVEAT cardinal:** n=2 simbolos -> el contraste esta CONFUNDIDO con la identidad del
  simbolo (no aisla volatilidad de "ser BTC vs ser ETH"). Es prior debil generador de
  hipotesis, NO evidencia (coherente con Pattern A heterogeneity, memoria). Se reporta APARTE
  del veredicto principal (brecha vs control + monotonia funding + consistencia BTC+ETH).
- Premisa "ETH mas volatil en 2023-2024" se VERIFICA primary-source en el reporte (vol
  realizada), no se asume.

**Disciplina ex-ante para la fase posterior (LINEA ROJA contra cherry-picking de universo):**
- "Volatil" = criterio objetivo medible ANTES de mirar supervivencia MFE (vol realizada
  media / ATR relativo / rango diario normalizado). Regla "vol ex-ante > umbral X", NO "los
  que mejor funcionaron".
- "No rug pulls" = criterios objetivos ex-ante (antiguedad listado, volumen sostenido, cap
  minima, profundidad de libro), NO "los que se que colapsaron".
- NUNCA seleccionar el universo por las reversiones historicas observadas (= cherry-picking
  de universo, identico al sesgo de los meses destructivos ya prohibido).

---

## 13. VEREDICTO DEL RUN 2 ANIOS (2026-06-29) — NEGATIVO ROBUSTO bajo falsador duro
Run 1.53h, 731 dias x 2 sym, prefix-invariance PASS (768 BTC / 903 ETH). Cascadas: BTC 1391
(neg 121 / mid 1232 / hi 38), ETH 1835 (neg 133 / mid 1662 / hi 40). **Ambos buckets hi>=30
y neg>=30 -> TEST PRIMARIO operativo en los dos** (Escenario A SI fue medible, ~la proyeccion
del smoke). Control 10x estratificado, CI cluster-8h.

**(1) CARDINAL — Expectativa neta NEGATIVA, sin edge vs aleatorio.** Retorno neto medio por
trade (slippage entrada peor-3s + 0.20% RT + stop -2%):
  - BTC: 1m -0.243% / 3m -0.226% / 5m -0.229%   (control -0.203/-0.201/-0.201%)
  - ETH: 1m -0.233% / 3m -0.232% / 5m -0.238%   (control -0.205/-0.206/-0.204%)
La cascada pierde ~-0.2% de media, IGUAL O PEOR que la entrada aleatoria. **MUERE bajo el
falsador.**

**(2) El "gap de supervivencia" NO se traduce en edge.** La supervivencia (% net>0) de la
cascada SI supera al control robustamente (23-27% vs 3-7%, CI cluster-8h excluye 0 en neg/mid
todos los horizontes) -> hay informacion microestructural real (los rebotes tras exhaustion
ocurren mas que al azar). PERO los perdedores de la cascada son MAYORES (cola -2% stop / MAE)
y compensan exactamente las ganancias extra -> la MEDIA neta no tiene edge. Mas ganadores
pequenos, mas perdedores grandes, suma cero-negativa.

**(3) Funding NO estratifica (la pregunta de la extension): REFUTADO.** Contraste primario
[gap(hi)-gap(neg)] CI cluster-8h incluye 0 en 5/6 celdas; unica que excluye 0+ = ETH-1min
(apenas, 0.0001), CONTRADICHA por BTC-1min (anti-monotono) y BTC-5min (significativamente EN
CONTRA, CI (-0.347,-0.176,-0.016)). Monotonia falla 4/6. Consistencia BTC+ETH FALLA. Las
salvaguardas (monotonia + consistencia + control simetrico) rechazan -> celda aislada = ruido.

**(4) DIAGNOSTICO volatilidad (APARTE del veredicto, n=2 confundido):** premisa verificada
primary-source (vol anualizada ETH 56% > BTC 48%; std 1min 0.081% > 0.070%; rango diario 4.64%
> 3.96%). PERO gap(cascada-control) en bucket mid IDENTICO: BTC +0.203 vs ETH +0.198 (3m),
+0.214 vs +0.213 (5m), ETH-BTC ~= 0 -> evidencia EN CONTRA de "mas volatil = mas edge". NO
motiva seleccion por volatilidad.

**VEREDICTO ASIMETRICO: NO robusto, cerrado por dias y $0.** La MFE no supera el falsador duro
(expectativa neta negativa, sin edge vs aleatorio); el funding no aporta filtro de regimen. El
backtest era FALSADOR: este "no" es valido (muere incluso sin cherry-picking, periodo continuo,
control simetrico, pre-registro congelado). La senal microestructural existe (gap supervivencia)
pero no basta para vencer la ejecucion pesimista. NO derecho a paper-trading. Decision
proyecto/bot = conversacion Ricardo aparte.

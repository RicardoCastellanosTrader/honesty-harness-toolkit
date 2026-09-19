# [NOMBRE DEL EXPERIMENTO] — Pre-registro (BORRADOR — PENDIENTE FREEZE)

**Fecha:** YYYY-MM-DD · **Autor:** [nombre] · **Estado:** BORRADOR → CONGELADO el YYYY-MM-DD por [director/autor]

> Plantilla extraída de los prerregistros reales del estudio (MFE, CS-MOM, campaña edge real, serie B1-B8 — ver DOI 10.5281/zenodo.21229492). Las 10 secciones siguientes son **obligatorias**, en este orden; el veredicto es SIEMPRE la última y se escribe al cierre sobre este mismo documento (trazabilidad ex-ante → resultado). Etiqueta cada sección con su estado inline: `[CONGELADA]`, `[PENDIENTE FREEZE]`, `[HECHAS]`.

## 0. Disciplina

- Este documento se CONGELA **antes** de ver ningún resultado. Después del freeze solo se admiten ENMIENDAS fechadas y justificadas (añadidas, nunca reescritas).
- Un experimento → una pregunta.
- El backtest es un **FALSADOR ASIMÉTRICO**: un "no" robusto cierra la hipótesis; un "sí" NO confirma edge — como mucho da derecho a paper-trading.
- Sandbox aislado, producción intacta, presupuesto declarado.

## 1. Hipótesis causal [CONGELAR]

Una sola hipótesis, con mecanismo causal explícito (POR QUÉ existiría este edge y quién paga por él). Si no puedes escribir el mecanismo, no tienes hipótesis: tienes un patrón.

## 2. Verificaciones primary-source [HACER ANTES DEL FREEZE]

Qué datos existen de verdad (rutas, cobertura, huecos), verificado leyéndolos — no de memoria ni de blogs. Lista aquí cada verificación con su evidencia.

## 3. Métrica cardinal a priori [CONGELAR]

UNA métrica decide (p. ej. expectativa neta por trade con CI95; Sharpe neto). Sin pivote posterior: si al ver resultados prefieres otra métrica, eso es una enmienda fechada ANTES de recalcular, o no vale.

## 4. Umbrales y falsador [CONGELAR]

Valores numéricos exactos: qué resultado mata la hipótesis, qué resultado daría derecho a paper-trading, y qué zona queda NO CONCLUYENTE. El intervalo de confianza va sobre el número que decide.

## 5. Placebos / controles / multiple-testing [CONGELAR]

Contra qué ruido se compara (GBM drift-0 con `engine/placebo_gen.py`, entradas aleatorias simétricas, benchmark trivial point-in-time…) y cómo se controla el número de intentos (Bonferroni/FDR si hay m>1 tests; declara m aquí).

## 6. Plan de análisis y parámetros a priori [CONGELAR]

Todos los parámetros fijados ANTES de correr (períodos, ventanas, costes — comisión realista taker, slippage si aplica —, universo, semillas). Lo que no esté aquí, no se puede tunear después.

## 7. Anti-leakage / point-in-time [CONGELAR]

Cómo se garantiza que ninguna decisión ve el futuro: gate de prefix-invariance, `merge_asof` backward para datos asíncronos, holdout intocado, `engine/leakage_gate.py` si aplica. Declara el presupuesto de holdout (qué datos quedan vírgenes y para qué).

## 8. Plan de cómputo y coste [CONGELAR]

Disco/CPU/GPU/horas estimadas (con factor de seguridad ×5-10 sobre tu proyección) y presupuesto monetario (idealmente $0). Si hay smoke de calibración previo, qué puede mirar y qué tiene PROHIBIDO mirar.

## 9. Qué NO se hará

Lista explícita anti-cherry-picking: no se probarán variantes no listadas, no se extenderá el período si el resultado no gusta, no se cambiará la métrica, no se reinterpretará un negativo como "casi".

## 10. VEREDICTO — [se escribe al cierre, contra los criterios de arriba]

Estructura del cierre (contrato de veredicto del estudio):
1. Metadatos del run + confirmación de que los gates ex-ante pasaron.
2. **La métrica cardinal contra el umbral congelado, lo primero.**
3. Hallazgos secundarios y salvaguardas (monotonía, consistencia, control).
4. Diagnósticos no-decisivos, marcados como tales.
5. **VEREDICTO ASIMÉTRICO** en negrita (POSITIVO→paper-trading / NEGATIVO / NO CONCLUYENTE).
6. La decisión de qué hacer con el proyecto se difiere a conversación aparte — no se toma dentro del acta.

---
*Secciones opcionales usadas en el estudio cuando aplican: smoke de calibración con resultado, enmiendas fechadas, libro mayor de holdouts, criterios de parada en 3 zonas.*

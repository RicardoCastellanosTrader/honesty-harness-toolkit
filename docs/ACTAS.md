# Actas y veredictos — cómo se cierran los experimentos

En el estudio original **no existe un generador automático de actas**: los
runners producen JSON/CSV de resultados, y el veredicto es un documento
redactado a mano contra los criterios pre-registrados. Es deliberado — el acto
de escribir el veredicto contra el prerregistro congelado es parte del
protocolo, no un formateo.

El contrato es:

1. El prerregistro se congela ANTES de ver resultados (plantilla:
   `docs/PREREGISTRO_TEMPLATE.md`).
2. El código computa su parte del veredicto y la vuelca a JSON (ver
   `sandboxes/cs_mom/cs_mom.py` → `results/curves.json` y
   `sandboxes/mfe/run_edge.py` → `edge_summary.json`).
3. El veredicto humano se escribe como ÚLTIMA SECCIÓN del propio prerregistro
   (no en archivo aparte), con la métrica cardinal primero y el rótulo
   **VEREDICTO ASIMÉTRICO** — así el documento conserva la trazabilidad
   completa ex-ante → resultado en un solo lugar.

Ejemplos reales (ambos terminaron en negativo, tal y como debe poder terminar
un experimento honesto): §13 de `sandboxes/mfe/MFE_FASE_EDGE_PREREGISTRO.md`
("NEGATIVO ROBUSTO") y §10 de `sandboxes/cs_mom/CS_MOM_PREREGISTRO.md`
("NO DIGNO").

Las actas y veredictos completos del estudio (campaña de edge real, estudio de
capacidad, Nivel 3, lista de cierre definitivo B1-B8) NO se copian a este
toolkit: viven sellados en el repositorio de evidencia —
DOI [10.5281/zenodo.21229492](https://doi.org/10.5281/zenodo.21229492) — y el
protocolo completo está en DOI
[10.5281/zenodo.21838807](https://doi.org/10.5281/zenodo.21838807).

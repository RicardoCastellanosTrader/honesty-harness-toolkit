# honesty-harness-toolkit (resumen en español)

Motor de backtesting walk-forward + protocolo de honestidad experimental (el
"Arnés de Honestidad"), extraídos del estudio *Anatomy of a Null Result: A
Pre-registered, Adversarially Audited Case Study of Retail Systematic Trading
in Crypto Perpetuals (2018-2026)* (Castellanos Macias, 2026) y publicados
para que terceros sometan sus propias estrategias a la misma maquinaria que
le devolvió un resultado **negativo** a su propio autor: 18 familias de
estrategias examinadas, todas negativas o por debajo del umbral, con los
siete veredictos principales meta-auditados por 39 agentes independientes.

**Qué es**: la mitad técnica (walk-forward por regímenes, bootstrap, filtro
de mesetas, placebo GBM drift-0, gates anti-leakage) elimina las formas
mecánicas de autoengañarse; la otra mitad es el protocolo (prerregistro
congelado con hash antes de mirar resultados, una métrica cardinal sin
pivote, falsador asimétrico, veredicto escrito contra los criterios sellados).
**Reduce el autoengaño; no lo garantiza.** Esto no es consejo de inversión:
la evidencia enlazada sugiere lo contrario para las familias de estrategias
estudiadas.

**Procedencia**: el motor y este toolkit fueron implementados por agentes de
IA bajo la dirección de un practicante del dominio que no programa, siguiendo
el protocolo del Arnés; el historial de commits lleva los trailers de
coautoría correspondientes.

**Rápido** (Python 3.12+): `pip install -r requirements.txt`, después los 5
comandos del README (sintético → toy → BTC real → walk-forward → rito
completo con veredicto contra el suelo de ruido). La lección medida: sobre
ruido puro, el mejor-de-61 promedia +83% y el mejor-de-1000 +138% (el listón
sube solo con mirar más); en BTC el ganador in-sample promete +74% de media y
entrega +6,5% forward; y el buy & hold (+1630%, contexto post-hoc) ganó al
"ganador" seleccionado (+566%). Figura: `examples/toy/ladder.png`. Ejemplo
sellado del rito: `examples/toy/PREREGISTRO_TOY.md` (en español, con su
sha256 y su recibo OpenTimestamps).

**Documentación**: guía de extensión (EN) en `docs/EXTENSION_GUIDE.md`;
plantillas de prerregistro en español (`docs/PREREGISTRO_TEMPLATE.md`) y en
inglés (`docs/PREREGISTRATION_TEMPLATE_EN.md`).

**Enlaces**: paper del estudio — Castellanos Macias, R. (2026), SSRN,
[https://ssrn.com/abstract=7085378](https://ssrn.com/abstract=7085378), DOI
[10.2139/ssrn.7085378](https://doi.org/10.2139/ssrn.7085378) · evidencia
sellada (repositorio `el-trading-no-existe-evidencia`) — DOI
[10.5281/zenodo.21229492](https://doi.org/10.5281/zenodo.21229492) ·
protocolo — DOI
[10.5281/zenodo.21838807](https://doi.org/10.5281/zenodo.21838807) ·
este toolkit — DOI
[10.5281/zenodo.22846067](https://doi.org/10.5281/zenodo.22846067).

**Licencia y soporte**: MIT, "as is". Bug reports bienvenidos; no se aceptan
feature requests (el motor se publica tal y como se usó en el estudio). Cómo
citar: `CITATION.cff`.

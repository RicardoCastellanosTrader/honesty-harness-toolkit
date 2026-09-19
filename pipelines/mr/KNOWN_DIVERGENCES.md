# pipelines/mr — Divergencias y limitaciones conocidas (documentadas, NO corregidas)

La tubería mean-reversion viaja **tal cual se usó en el estudio** (decisión del
director, Sesión 2). Corregir cualquiera de estos puntos rompería la fidelidad
con la evidencia publicada. El principio del motor es "kernel = verdad
operacional" (la rentabilidad simulada del estudio se construyó sobre ESTE
comportamiento, no sobre el diseño ideal).

## 1. Trailing stop SIEMPRE activo (sin gate `use_ts`)

- **MR**: el trailing on-close 0,5% se aplica incondicionalmente —
  `mean_reversion_kernel.py:278-287` (en este directorio).
- **TF**: el mismo trailing está condicionado al bit 23 `use_ts` del config —
  `engine/lab_historico_numba_v8_3.py:1528-1538` (gate en `:1528`).

El config MR (17 bits) no tiene bit `use_ts`: ninguna config MR puede apagar el
trailing. La mecánica de los 4 stops (SL inicial 3% desde mecha, TS on-close
0,5%, SL emergency 5% intrabar, trigger `close < sl_level`) es por lo demás
idéntica entre TF y MR. Si añades una tercera tubería, decide y DOCUMENTA esta
elección explícitamente en tu kernel.

## 2. Divergencia Hidden vs Pine v7.25 (intencional)

Los bits 12-13 del config MR implementan la divergencia Hidden **corregida**
(`mean_reversion_kernel.py:5-13`, ramas `:220-237`); el Pine MR histórico
v7.25 tenía la Hidden sin el fix de inversión. Es decisión de diseño madurada
del estudio: el kernel es la verdad operacional, el Pine es referencia
histórica.

## 3. `audit_mr_fidelity_sei.py` requiere una réplica viva que NO se distribuye

El script de auditoría de fidelidad compara el kernel MR contra el brain del
bot de producción (`live/brain_engine.py`), que este toolkit NO incluye. Se
distribuye como **documentación ejecutable del protocolo de fidelidad**
(qué métricas se comparan: PnL con tolerancia <0,01, trades y cancels exactos,
hash SHA256 del kernel para detectar cambios sin recertificar — ver
`audit_mr_fidelity_sei.py:39-55` y `:202-216`), no como script ejecutable aquí.
Su import falla sin `live/` — es esperado.

## 4. Dependencias de import del motor

Los tres módulos MR usan imports planos del repo de investigación:
`from master import CONFIG` (`mean_reversion_kernel.py:27`,
`mean_reversion_features.py:35`), `from lab_historico_numba_v8_3 import (...)`
(`mean_reversion_features.py:23`) y `from regime_walk_forward import (...)`
(`mean_reversion_walk_forward.py:26-37`). Ejecuta con
`PYTHONPATH=engine;pipelines/mr` (ver README) — no se han reescrito los
imports (regla: copiar, no refactorizar).

## 5. `test_path_gamma_mr.py` — test_4 espera el layout del repo original

El test standalone pasa **6/7** en este layout. El único FAIL (`test_4_mr_audit_
hash_regen_no_warn`) busca `audit_mr_fidelity_sei.py` en la raíz del repo
original (construye la ruta como `parent(parent(__file__))/audit_...`, que aquí
resuelve a `pipelines/` en vez de `pipelines/mr/`). Es un artefacto de layout,
no una regresión del kernel: los 6 tests de semántica del kernel (backward
compat, per-trade tracking, splits de señales) PASAN, y el hash SHA256 del
kernel copiado coincide con el certificado del estudio (verificado en el
smoke de importación de la Sesión 2). No se corrige por la regla
"copiar, no refactorizar".

## 6. Sin eje de presets

MR no consume `presets/*.csv`: sus medias son fijas (estilo Tenkan-9) y sus
features se pregeneran a `data_cache/{SYM}_mean_reversion.npz` con
`mean_reversion_features.py` (`precalculate_mean_reversion`). El eje preset ×
histéresis es exclusivo de la tubería TF.

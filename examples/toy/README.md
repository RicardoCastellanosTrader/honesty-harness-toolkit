# Ejemplo toy — recorrido mínimo

La tubería vive en `pipelines/toy/` (features → kernel → run_on_slice →
entrypoint, el contrato completo del motor en ~350 líneas legibles). Aquí solo
los comandos, desde la raíz del toolkit:

```bash
# 1) Muestra sintética GBM drift-0 (SIN edge por construcción), con el
#    generador de placebo canónico del estudio:
python examples/data/make_synthetic.py

# 2) Toy sobre la sintética (<2 min CPU; la primera ejecución compila el JIT):
python pipelines/toy/toy_pipeline.py --data examples/data/TOYGBM1USDT_1h.parquet

# 3) Toy sobre datos reales (descarga solo ese símbolo, ~1 min):
python pipelines/toy/toy_pipeline.py --symbol BTC/USDT --download
```

Qué mirar: en el paso 2 habrá configs con PnL positivo **sobre ruido puro** —
ese es el noise floor de quedarse con el máximo de 61 intentos. El paso 3 solo
significa algo comparado contra ese suelo. El walk-forward de la toy y el
veredicto formal contra placebo llegan en la Sesión 3.

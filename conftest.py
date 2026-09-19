# conftest.py — raíz del toolkit.
# Los módulos del motor usan imports planos entre sí (import master,
# from regime_walk_forward import ..., from mean_reversion_kernel import ...),
# heredados del repo original de investigación donde todo vivía en la raíz.
# Este conftest añade los directorios del toolkit a sys.path para pytest.
# Para ejecución de scripts fuera de pytest: exporta
#   PYTHONPATH=engine;pipelines/mr;pipelines/toy   (Windows)
#   PYTHONPATH=engine:pipelines/mr:pipelines/toy   (Linux/macOS)
# o usa los entrypoints (toy_pipeline.py, make_synthetic.py), que se
# auto-configuran.
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ('engine', os.path.join('pipelines', 'mr'), os.path.join('pipelines', 'toy')):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

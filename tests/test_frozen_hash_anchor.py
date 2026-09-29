"""Regresión del issue #1: frozen_hash debe anclar en la LÍNEA del marcador
FREEZE-BOUNDARY (línea que empieza por '<!-- FREEZE-BOUNDARY'), no en la
primera aparición textual del literal — una mención en prosa antes del
marcador no debe cortar el hash. También fija que el sidecar existente de
PREREGISTRO_TOY.md sigue verificando con la función corregida.
"""
import hashlib
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'engine'))
sys.path.insert(0, os.path.join(_ROOT, 'pipelines', 'toy'))
sys.path.insert(0, os.path.join(_ROOT, 'examples', 'toy'))

import placebo_verdict as pv


def test_prose_mention_before_marker_does_not_cut(tmp_path):
    doc = (
        "# Prereg de prueba\n"
        "\n"
        "La convencion usa el marcador `<!-- FREEZE-BOUNDARY` inclusive; esta\n"
        "mencion en prosa NO debe cortar el hash.\n"
        "\n"
        "## Cuerpo congelado\n"
        "contenido\n"
        "\n"
        "<!-- FREEZE-BOUNDARY: todo lo anterior, incluida esta linea. -->\n"
        "\n"
        "## Veredicto posterior\n"
        "no congelado\n"
    )
    p = tmp_path / 'PREREG_TEST.md'
    p.write_bytes(doc.encode('utf-8'))
    raw = doc.encode('utf-8')
    marker_line_start = raw.index(b'\n<!-- FREEZE-BOUNDARY') + 1
    end = raw.index(b'\n', marker_line_start) + 1
    expected = hashlib.sha256(raw[:end]).hexdigest()
    assert pv.frozen_hash(str(p)) == expected
    # y el corte NO es el de la primera aparicion textual del literal
    naive_idx = raw.find(b'<!-- FREEZE-BOUNDARY')
    naive_end = raw.index(b'\n', naive_idx) + 1
    naive = hashlib.sha256(raw[:naive_end]).hexdigest()
    assert pv.frozen_hash(str(p)) != naive


def test_marker_at_first_byte(tmp_path):
    doc = "<!-- FREEZE-BOUNDARY: linea unica -->\nresto\n"
    p = tmp_path / 'PREREG_EDGE.md'
    p.write_bytes(doc.encode('utf-8'))
    raw = doc.encode('utf-8')
    end = raw.index(b'\n') + 1
    assert pv.frozen_hash(str(p)) == hashlib.sha256(raw[:end]).hexdigest()


def test_existing_toy_sidecar_still_verifies():
    """El hash del prerregistro toy sellado no cambia con la corrección."""
    h = pv.frozen_hash()
    assert h == '0519812ac68206162c585440945065b68e454330688bc38af59bb8e090772bb0'
    with open(pv.SIDECAR, encoding='utf-8') as f:
        sidecar = f.read()
    assert h in sidecar

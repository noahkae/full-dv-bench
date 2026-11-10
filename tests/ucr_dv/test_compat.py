"""The team runs Python 3.8 (ROS 2 Galactic / Ubuntu 20.04): keep every file 3.8-parseable,
and keep the algorithm core free of Qt so the ROS node doesn't need a display stack."""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _py_files():
    for top in ('ucr_dv', 'ros', 'fsds_lite'):
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                if f.endswith('.py'):
                    yield os.path.join(d, f)


def test_sources_parse_as_python_3_8():
    bad = []
    for path in _py_files():
        with open(path) as f:
            src = f.read()
        try:
            ast.parse(src, path, feature_version=(3, 8))
        except SyntaxError as e:
            bad.append('{}:{} {}'.format(path, e.lineno, e.msg))
    assert not bad, '\n'.join(bad)


def test_core_does_not_import_qt():
    code = ('import sys; sys.path.insert(0, {!r}); import ucr_dv.pipeline, ucr_dv.bench.session, '
            'ucr_dv.backends.timeless, ucr_dv.deploy; '
            'from ucr_dv.core import registry; registry.entries("planner"); '
            'print("PyQt5" in sys.modules)').format(ROOT)
    out = subprocess.check_output([sys.executable, '-c', code]).decode().strip()
    assert out == 'False'

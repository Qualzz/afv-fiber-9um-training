"""Run the archived synthetic geometry checks without initializing its data loader."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / 'original/fiber-centre-compression-v6'


def main():
    spec = importlib.util.spec_from_file_location('original_geometry', V6 / 'geometry.py')
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    source = ast.parse((V6 / 'prepare.py').read_text())
    check = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'geometry_checks')
    isolated = ast.Module(body=[check], type_ignores=[])
    namespace = {'np': np, 'Compaction': geometry.Compaction}
    exec(compile(isolated, str(V6 / 'prepare.py'), 'exec'), namespace)
    print(json.dumps(namespace['geometry_checks'](), indent=2))


if __name__ == '__main__':
    main()

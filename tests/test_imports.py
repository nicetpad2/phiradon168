import os
import ast

FILES = [
    'src/config.py',
    'src/data_loader.py',
    'src/features.py',
    'src/strategy.py',
    'src/main.py',
]

def test_parseable():
    for path in FILES:
        with open(path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())

import os
import ast

FILES = [
    'src/config.py',
    'src/data_loader.py',
    'src/features.py',
    'src/cooldown_utils.py',
    'src/strategy.py',
    'strategy/entry_rules.py',
    'strategy/exit_rules.py',
    'src/main.py',
]

def test_parseable():
    for path in FILES:
        with open(path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())

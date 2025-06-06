import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

import placeholder


def test_add_and_multiply():
    assert placeholder.add(2, 3) == 5
    assert placeholder.multiply(4, 5) == 20

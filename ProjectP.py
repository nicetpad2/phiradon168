import sys
import os

# [Patch] Set up import paths for local modules
REPO_ROOT = os.path.dirname(__file__)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.main import main

def custom_helper_function():
    """Stubbed helper for tests."""
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n(Stopped) การทำงานถูกยกเลิกโดยผู้ใช้.")

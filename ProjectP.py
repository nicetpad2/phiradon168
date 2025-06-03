import sys
import os

# [Patch v5.0.14] Ensure proper import path for src package
REPO_ROOT = os.path.dirname(__file__)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.main import main


def custom_helper_function():
    """Stubbed helper for tests."""
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n(Stopped) การทำงานถูกยกเลิกโดยผู้ใช้.")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.main import main


def custom_helper_function():
    """Stubbed helper for tests."""
    return True

if __name__ == '__main__':
    main()

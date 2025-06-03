"""Bootstrap script for running the main entry point."""

from src.main import main

def custom_helper_function():
    """Stubbed helper for tests."""
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n(Stopped) การทำงานถูกยกเลิกโดยผู้ใช้.")

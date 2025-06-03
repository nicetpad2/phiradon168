"""Bootstrap script for running the main entry point."""

from src.config import logger
import sys
from src.main import main

def custom_helper_function():
    """Stubbed helper for tests."""
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n(Stopped) การทำงานถูกยกเลิกโดยผู้ใช้.")
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)

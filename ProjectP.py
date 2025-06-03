"""Bootstrap script for running the main entry point."""

from src.config import logger
import sys
import logging

# [Patch] Initialize pynvml for GPU status detection
try:
    import pynvml

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None
    nvml_handle = None
except Exception:  # pragma: no cover - NVML failure fallback
    nvml_handle = None

from src.main import main


def custom_helper_function():
    """Stubbed helper for tests."""
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(Stopped) การทำงานถูกยกเลิกโดยผู้ใช้.")
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)
    else:
        # [Patch v5.0.23] Respect USE_GPU_ACCELERATION flag when logging GPU status
        main_mod = sys.modules.get("src.main")
        use_gpu = getattr(main_mod, "USE_GPU_ACCELERATION", False)
        if use_gpu and "pynvml" in globals() and nvml_handle:
            try:
                pynvml.nvmlShutdown()
                logging.info("GPU resources released")
            except Exception as e:
                logging.warning(f"Failed to shut down NVML: {e}")
        else:
            logging.info("GPU not available, running on CPU")

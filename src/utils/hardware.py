import logging

# [Patch v5.5.14] GPU detection helper

def has_gpu() -> bool:
    """Return True if a CUDA-capable GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception as e:
        logging.debug("GPU detection failed: %s", e)
        return False

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


# [Patch v5.7.3] Estimate resource-based parameters with safe fallbacks
def estimate_resource_plan(default_folds=5, default_batch=32):
    """Return a simple plan of ``n_folds`` and ``batch_size``.

    If ``psutil`` or ``torch`` are not available, defaults are returned.
    """
    gpu_name = "Unknown"
    if has_gpu():
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as e:
            logging.debug("GPU name lookup failed: %s", e)
    try:
        import psutil
        total_gb = psutil.virtual_memory().total / 1024 ** 3
    except Exception as e:
        logging.warning("psutil unavailable; using default resource plan: %s", e)
        return {
            "n_folds": default_folds,
            "batch_size": default_batch,
            "gpu": gpu_name,
        }

    n_folds = default_folds if total_gb >= 4 else max(2, default_folds - 2)
    batch_size = default_batch if total_gb >= 8 else max(8, default_batch // 2)
    return {"n_folds": int(n_folds), "batch_size": int(batch_size), "gpu": gpu_name}

import json
import os
import logging

logger = logging.getLogger(__name__)


def get_resource_plan() -> dict:
    """Return basic resource information."""
    try:
        import psutil

        mem = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception as e:
        logger.debug("psutil not available: %s", e)
        mem = 0.0
    cpu = os.cpu_count() or 0
    return {"cpu_count": cpu, "total_memory_gb": mem}


def save_resource_plan(output_dir: str) -> None:
    """Save resource plan JSON to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "resource_plan.json")
    plan = get_resource_plan()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f)
    logger.info("Resource plan saved to %s", path)

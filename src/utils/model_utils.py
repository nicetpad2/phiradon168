import os
import logging
import urllib.request

# [Patch v5.5.0] Utility to download model files if missing

def download_model_if_missing(model_path, url_env):
    """Download model from URL specified in environment variable if missing."""
    if os.path.exists(model_path):
        return
    url = os.getenv(url_env)
    if not url:
        return
    try:
        logging.info(f"(Info) Downloading model from {url}...")
        urllib.request.urlretrieve(url, model_path)
        logging.info("(Success) Model downloaded.")
    except Exception as e:
        logging.error(f"(Error) Failed to download model from {url}: {e}")

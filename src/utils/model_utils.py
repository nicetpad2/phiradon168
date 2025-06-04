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

# [Patch v5.6.0] Utility to download feature list files if missing
def download_feature_list_if_missing(features_path, url_env):
    """Download feature list from URL specified in environment variable if missing."""
    if os.path.exists(features_path):
        return
    url = os.getenv(url_env)
    if not url:
        return
    try:
        logging.info(f"(Info) Downloading feature list from {url}...")
        urllib.request.urlretrieve(url, features_path)
        logging.info("(Success) Feature list downloaded.")
    except Exception as e:
        logging.error(f"(Error) Failed to download feature list from {url}: {e}")

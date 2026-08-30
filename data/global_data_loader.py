import gc
import os

import numpy as np

from data.data_loader_haanglanden import load_data_haaglanden
from data.data_loader_sleep_edfx import load_data_sleep_edfx

HPO_SEARCH_SAMPLE_SIZE = 8000
HPO_SAMPLE_RANDOM_STATE = 42

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _cache_path(name, n_samples, random_state):
    return os.path.join(CACHE_DIR, f"{name}_{n_samples}_{random_state}.npz")


def _load_cached(name, loader, n_samples, random_state):
    """Decode once, reuse across all ten optimizer runs.

    Every optimizer script is a separate process, so without this each one
    re-decodes the full ~21 GB of EDF from scratch.
    """
    path = _cache_path(name, n_samples, random_state)

    if os.path.exists(path):
        print(f"[{name}] Loading cached arrays from {path}")
        with np.load(path) as cached:
            return cached["X"], cached["y"]

    X, y = loader(n_samples=n_samples, random_state=random_state)

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = path + ".tmp.npz"
    np.savez(tmp_path, X=X, y=y)
    os.replace(tmp_path, path)
    print(f"[{name}] Cached arrays to {path}")

    return X, y


def get_data_all_datasets(subsample=True):
    n_samples = HPO_SEARCH_SAMPLE_SIZE if subsample else None

    haaglanden = _load_cached(
        "haaglanden", load_data_haaglanden, n_samples, HPO_SAMPLE_RANDOM_STATE
    )
    gc.collect()

    sleep_edfx = _load_cached(
        "sleep_edfx", load_data_sleep_edfx, n_samples, HPO_SAMPLE_RANDOM_STATE
    )
    gc.collect()

    return haaglanden, sleep_edfx

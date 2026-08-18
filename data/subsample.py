import numpy as np


def subsample_stratified(X, y, n_samples, random_state=42):
    """
    Class-balanced subsample used for HPO search across all algorithms.
    Returns X, y unchanged if n_samples is None or >= len(X).
    """
    if n_samples is None or n_samples >= len(X):
        return X, y

    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    per_class = max(1, n_samples // len(classes))

    indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        take = min(per_class, len(cls_indices))
        chosen = rng.choice(cls_indices, size=take, replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    rng.shuffle(indices)

    return X[indices], y[indices]
"""Shared EDF loading helpers.

Both dataset loaders follow the same two-pass shape:

  Pass 1 (cheap)  -- read only the EDF *header* for the recording length and the
                     annotation file for the labels. No signal is decoded.
  Select          -- pick a class-balanced subset of epochs across all files.
  Pass 2 (costly) -- decode only the files that contributed selected epochs, and
                     keep only those epoch rows.

The old implementation decoded every recording twice and materialised the whole
dataset (~9 GB for sleep-cassette) before subsampling to 8000 epochs, which
exceeded this machine's commit limit during ``normalize_epochs``.
"""

import gc
import os
import warnings

import mne
import numpy as np

TARGET_SFREQ = 100
SAMPLES_PER_EPOCH = 30 * TARGET_SFREQ

# The two datasets use different scoring conventions:
#   Sleep-EDF   (R&K)  -- "Sleep stage 1/2/3/4"
#   Haaglanden  (AASM) -- "Sleep stage N1/N2/N3"
# Both are mapped onto the same 5 classes (R&K stages 3+4 merge into N3).
STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage N1": 1,
    "Sleep stage 2": 2,
    "Sleep stage N2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}

# Epochs with no usable stage ("Sleep stage ?", "Movement time", lights on/off
# markers, gaps) are marked with this and excluded from selection.
UNSCORED = -1

warnings.filterwarnings("ignore", message="Channels contain different highpass filters")
warnings.filterwarnings("ignore", message="Channels contain different lowpass filters")
# EDF headers in both datasets carry inconsistent filter metadata; harmless here
# since we do not rely on the reported cutoffs, but it fires once per file read.
warnings.filterwarnings("ignore", message="Highpass cutoff frequency")


def count_epochs_from_header(path):
    """Number of 30s epochs in an EDF, without decoding any signal data."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    duration_seconds = raw.n_times / raw.info["sfreq"]
    return int(duration_seconds // 30)


def load_hypnogram(path, n_epochs=None):
    """Labels indexed by epoch number, using annotation onset for placement.

    Placing by onset rather than appending is essential: annotations that carry
    no usable stage (e.g. "Movement time" mid-recording, or Haaglanden's
    "Lights off@@..." markers) must not shift every later label out of
    alignment with its EEG epoch. Unscored slots get UNSCORED.
    """
    annotations = mne.read_annotations(path)

    placed = {}
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        # mne may hand back numpy str scalars; normalise before lookup.
        desc = str(desc)
        if desc not in STAGE_MAPPING:
            continue
        stage = STAGE_MAPPING[desc]
        start_epoch = int(round(float(onset) / 30.0))
        span = int(float(duration) // 30)
        for k in range(span):
            placed[start_epoch + k] = stage

    if n_epochs is None:
        n_epochs = (max(placed) + 1) if placed else 0

    labels = np.full(n_epochs, UNSCORED, dtype=np.int64)
    for epoch_index, stage in placed.items():
        if 0 <= epoch_index < n_epochs:
            labels[epoch_index] = stage

    return labels


def load_eeg(path, channels):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    raw.pick(channels)
    raw.load_data()
    raw.resample(TARGET_SFREQ)
    return raw.get_data().astype(np.float32)


def create_epochs(eeg):
    n_epochs = eeg.shape[1] // SAMPLES_PER_EPOCH
    eeg = eeg[:, :n_epochs * SAMPLES_PER_EPOCH]
    epochs = eeg.reshape(eeg.shape[0], n_epochs, SAMPLES_PER_EPOCH)
    return np.transpose(epochs, (1, 0, 2))


def normalize_epochs(X):
    """Per-epoch, per-channel z-score, done in place to avoid extra copies."""
    if len(X) == 0:
        return X
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    X -= mean
    X /= (std + 1e-8)
    return X


def warn_about_orphans(psg_files, hyp_files, dataset_label):
    """Report recordings that are missing their partner file instead of silently
    dropping them -- an incomplete download used to be invisible."""
    psg_only = sorted(set(psg_files) - set(hyp_files))
    hyp_only = sorted(set(hyp_files) - set(psg_files))

    if psg_only:
        print(
            f"[{dataset_label}] WARNING: {len(psg_only)} recording(s) have a PSG "
            f"file but no annotation file and will be skipped: "
            f"{', '.join(psg_only[:10])}{' ...' if len(psg_only) > 10 else ''}"
        )
    if hyp_only:
        print(
            f"[{dataset_label}] WARNING: {len(hyp_only)} recording(s) have an "
            f"annotation file but no PSG file and will be skipped: "
            f"{', '.join(hyp_only[:10])}{' ...' if len(hyp_only) > 10 else ''}"
        )


def select_epoch_indices(labels_per_file, n_samples, random_state):
    """Choose a class-balanced set of epochs across every file.

    ``labels_per_file`` is a list of per-file label arrays. Returns a list of
    per-file sorted index arrays -- the epochs to actually decode.

    Mirrors ``subsample_stratified``: an equal quota per class, capped by
    availability, then shuffled.
    """
    file_ids = np.concatenate([
        np.full(len(labels), i, dtype=np.int32)
        for i, labels in enumerate(labels_per_file)
    ])
    epoch_ids = np.concatenate([
        np.arange(len(labels), dtype=np.int32) for labels in labels_per_file
    ])
    all_labels = np.concatenate(labels_per_file)

    # Epochs with no usable stage are never eligible.
    scored = np.where(all_labels != UNSCORED)[0]
    n_scored = len(scored)
    if n_scored == 0:
        raise ValueError(
            "No scored epochs found -- every annotation was unrecognised. "
            f"Known stage names: {sorted(STAGE_MAPPING)}"
        )

    if n_samples is None or n_samples >= n_scored:
        chosen = scored
    else:
        rng = np.random.RandomState(random_state)
        classes = np.unique(all_labels[scored])
        per_class = max(1, n_samples // len(classes))

        picked = []
        for cls in classes:
            cls_indices = scored[all_labels[scored] == cls]
            take = min(per_class, len(cls_indices))
            picked.append(rng.choice(cls_indices, size=take, replace=False))
        chosen = np.concatenate(picked)
        rng.shuffle(chosen)

    per_file = [[] for _ in labels_per_file]
    for flat_index in chosen:
        per_file[file_ids[flat_index]].append(epoch_ids[flat_index])

    return [np.sort(np.array(idx, dtype=np.int32)) for idx in per_file]


def load_dataset(pairs, channels, dataset_label, n_samples, random_state):
    """Two-pass loader shared by both datasets.

    ``pairs`` is a list of ``(psg_path, annotation_path)`` tuples.
    Returns ``(X, y)`` with X shaped ``(n_selected, len(channels), 3000)``.
    """
    n_channels = len(channels)

    # --- Pass 1: headers + annotations only, no signal decoding -------------
    labels_per_file = []
    for i, (psg_path, hyp_path) in enumerate(pairs, start=1):
        print(f"[{dataset_label}] Scanning {i}/{len(pairs)}: {os.path.basename(psg_path)}")
        n_eeg_epochs = count_epochs_from_header(psg_path)
        labels = load_hypnogram(hyp_path, n_epochs=n_eeg_epochs)
        labels_per_file.append(labels)

    total_epochs = sum(len(labels) for labels in labels_per_file)
    print(f"[{dataset_label}] Total epochs available: {total_epochs}")

    # --- Select which epochs we actually want -------------------------------
    selected = select_epoch_indices(labels_per_file, n_samples, random_state)
    n_selected = sum(len(idx) for idx in selected)
    files_needed = sum(1 for idx in selected if len(idx) > 0)
    print(
        f"[{dataset_label}] Selected {n_selected} epochs "
        f"from {files_needed}/{len(pairs)} recordings"
    )

    X = np.empty((n_selected, n_channels, SAMPLES_PER_EPOCH), dtype=np.float32)
    y = np.empty((n_selected,), dtype=np.int64)

    # --- Pass 2: decode only what we need -----------------------------------
    offset = 0
    for i, ((psg_path, hyp_path), indices) in enumerate(zip(pairs, selected), start=1):
        if len(indices) == 0:
            continue

        print(
            f"[{dataset_label}] Loading {i}/{len(pairs)}: "
            f"{os.path.basename(psg_path)} ({len(indices)} epochs)"
        )
        eeg = load_eeg(psg_path, channels)
        epochs = create_epochs(eeg)

        # The header-derived count and the decoded count agree in practice, but
        # a one-epoch float-rounding disagreement on any of the ~300 files would
        # otherwise abort a multi-hour run with an IndexError.
        usable = indices[indices < epochs.shape[0]]
        if len(usable) != len(indices):
            print(
                f"[{dataset_label}] NOTE: {os.path.basename(psg_path)} decoded to "
                f"{epochs.shape[0]} epochs but the header implied "
                f"{len(labels_per_file[i - 1])}; dropping "
                f"{len(indices) - len(usable)} out-of-range epoch(s)."
            )

        n = len(usable)
        X[offset:offset + n] = epochs[usable]
        y[offset:offset + n] = labels_per_file[i - 1][usable]
        offset += n

        del eeg, epochs
        gc.collect()

    # Trim in the (unexpected) event that some epochs were dropped above.
    if offset != len(X):
        print(f"[{dataset_label}] Trimming {len(X) - offset} unfilled row(s)")
        X = X[:offset]
        y = y[:offset]

    X = normalize_epochs(X)

    counts = np.bincount(y, minlength=5)
    stage_names = ["W", "N1", "N2", "N3", "REM"]
    breakdown = ", ".join(f"{n}={c}" for n, c in zip(stage_names, counts))

    print(f"\n[{dataset_label}] Finished loading")
    print(f"[{dataset_label}] X shape: {X.shape}")
    print(f"[{dataset_label}] y shape: {y.shape}")
    print(f"[{dataset_label}] class balance: {breakdown}")

    if (counts == 0).any():
        missing = [n for n, c in zip(stage_names, counts) if c == 0]
        print(
            f"[{dataset_label}] WARNING: no epochs for stage(s) "
            f"{', '.join(missing)} -- check STAGE_MAPPING against this "
            f"dataset's annotation names."
        )

    return X, y

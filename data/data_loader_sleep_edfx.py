import os
import gc
import warnings
import mne
import numpy as np

warnings.filterwarnings("ignore", message="Channels contain different highpass filters")
warnings.filterwarnings("ignore", message="Channels contain different lowpass filters")

STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
TARGET_SFREQ = 100
SAMPLES_PER_EPOCH = 30 * TARGET_SFREQ


def get_matched_pairs(data_dir):
    psg_files = {}
    hyp_files = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            # Match on the first 6 chars (subject+night code).
            # PSG and Hypnogram files for the same recording do NOT
            # share a full identical prefix -- e.g. SC4001E0-PSG.edf
            # pairs with SC4001EC-Hypnogram.edf, not SC4001E0-Hypnogram.edf.
            if file.endswith("-PSG.edf"):
                key = file[:6]
                psg_files[key] = path
            elif file.endswith("-Hypnogram.edf"):
                key = file[:6]
                hyp_files[key] = path
    return [(psg_files[key], hyp_files[key]) for key in psg_files if key in hyp_files]


def load_hypnogram(path):
    annotations = mne.read_annotations(path)
    stages = []
    for desc, duration in zip(annotations.description, annotations.duration):
        if desc in STAGE_MAPPING:
            stages.extend([STAGE_MAPPING[desc]] * int(duration // 30))
    return np.array(stages)


def load_eeg(path):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    raw.pick(EEG_CHANNELS)
    raw.load_data()
    raw.resample(TARGET_SFREQ)
    return raw.get_data().astype(np.float32)


def create_epochs(eeg):
    n_epochs = eeg.shape[1] // SAMPLES_PER_EPOCH
    eeg = eeg[:, :n_epochs * SAMPLES_PER_EPOCH]
    epochs = eeg.reshape(eeg.shape[0], n_epochs, SAMPLES_PER_EPOCH)
    return np.transpose(epochs, (1, 0, 2))


def normalize_epochs(X):
    if len(X) == 0:
        return X
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return ((X - mean) / (std + 1e-8)).astype(np.float32)

import os
from dotenv import load_dotenv

load_dotenv()

DATA_SLEEP_EDFX = os.getenv("DATA_SLEEP_EDFX")

def load_data_sleep_edfx():
    data_dir = DATA_SLEEP_EDFX
    if not data_dir:
        raise RuntimeError(
            "DATA_SLEEP_EDFX is not set. Copy .env.example to .env and "
            "point DATA_SLEEP_EDFX at your dataset directory."
        )
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(
            f"DATA_SLEEP_EDFX points at {data_dir!r}, which is not a directory."
        )
    pairs = get_matched_pairs(data_dir)
    print(f"Found {len(pairs)} PSG/Hypnogram pairs")

    if len(pairs) == 0:
        raise ValueError(
            f"No matched PSG/Hypnogram pairs found in {data_dir} -- "
            f"check that filenames follow the '<key>-PSG.edf' / "
            f"'<key>-Hypnogram.edf' convention and that 'key' (first "
            f"6 chars) actually matches between the two files."
        )

    n_channels = len(EEG_CHANNELS)

    epoch_counts = []
    for i, (psg_path, hyp_path) in enumerate(pairs, start=1):
        print(f"Counting {i}/{len(pairs)}: {os.path.basename(psg_path)}")
        eeg = load_eeg(psg_path)
        n_eeg_epochs = eeg.shape[1] // SAMPLES_PER_EPOCH
        labels = load_hypnogram(hyp_path)
        n = min(n_eeg_epochs, len(labels))
        epoch_counts.append(n)
        del eeg, labels
        gc.collect()

    total_epochs = sum(epoch_counts)
    print(f"Total epochs across all files: {total_epochs}")

    X = np.empty((total_epochs, n_channels, SAMPLES_PER_EPOCH), dtype=np.float32)
    y = np.empty((total_epochs,), dtype=np.int64)

    offset = 0
    for i, ((psg_path, hyp_path), n) in enumerate(zip(pairs, epoch_counts), start=1):
        print(f"Loading {i}/{len(pairs)}: {os.path.basename(psg_path)}")
        eeg = load_eeg(psg_path)
        epochs = create_epochs(eeg)
        labels = load_hypnogram(hyp_path)

        X[offset:offset + n] = epochs[:n]
        y[offset:offset + n] = labels[:n]
        offset += n

        del eeg, epochs, labels
        gc.collect()

    X = normalize_epochs(X)

    print("\nFinished loading")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y
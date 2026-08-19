import os
import gc
import warnings
import mne
import numpy as np


warnings.filterwarnings(
    "ignore",
    message="Channels contain different highpass filters"
)

warnings.filterwarnings(
    "ignore",
    message="Channels contain different lowpass filters"
)


STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


EEG_CHANNELS = [
    "EEG C4-M1",
    "EEG O2-M1",
]


TARGET_SFREQ = 100
SAMPLES_PER_EPOCH = 30 * TARGET_SFREQ


def get_matched_pairs(data_dir):
    psg_files = {}
    hyp_files = {}

    for file in os.listdir(data_dir):

        if file.endswith(".edf") and "_sleepscoring" not in file:
            key = file.replace(".edf", "")
            psg_files[key] = os.path.join(data_dir, file)

        elif file.endswith("_sleepscoring.edf"):
            key = file.replace("_sleepscoring.edf", "")
            hyp_files[key] = os.path.join(data_dir, file)

    return [
        (psg_files[key], hyp_files[key])
        for key in psg_files
        if key in hyp_files
    ]


def load_hypnogram(path):
    annotations = mne.read_annotations(path)
    labels = []

    for description, duration in zip(annotations.description, annotations.duration):
        if description in STAGE_MAPPING:
            stage = STAGE_MAPPING[description]
            epochs = int(duration // 30)
            labels.extend([stage] * epochs)

    return np.array(labels)


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
    epochs = np.transpose(epochs, (1, 0, 2))
    return epochs


def normalize_epochs(X):
    if len(X) == 0:
        return X
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return ((X - mean) / (std + 1e-8)).astype(np.float32)

import os
from dotenv import load_dotenv

load_dotenv()

DATA_HAANGLANDEN = os.getenv("DATA_HAANGLANDEN")

def load_data_haaglanden():
    data_dir = DATA_HAANGLANDEN
    if not data_dir:
        raise RuntimeError(
            "DATA_HAANGLANDEN is not set. Copy .env.example to .env and "
            "point DATA_HAANGLANDEN at your dataset directory."
        )
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(
            f"DATA_HAANGLANDEN points at {data_dir!r}, which is not a directory."
        )
    pairs = get_matched_pairs(data_dir)
    print(f"Found {len(pairs)} PSG/Hypnogram pairs")

    if len(pairs) == 0:
        raise ValueError(f"No matched PSG/Hypnogram pairs found in {data_dir}")

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


if __name__ == "__main__":
    X, y = load_data_haaglanden()
    print("\nExample:")
    print("First epoch shape:", X[0].shape)
    print("First label:", y[0])
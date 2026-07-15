import os
import mne
import numpy as np


STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


def get_matched_pairs(data_dir):
    psg_files = {}
    hyp_files = {}

    for f in os.listdir(data_dir):
        if f.endswith("-PSG.edf"):
            psg_files[f[:6]] = os.path.join(data_dir, f)
        elif f.endswith("-Hypnogram.edf"):
            hyp_files[f[:6]] = os.path.join(data_dir, f)

    return [
        (psg_files[k], hyp_files[k])
        for k in psg_files
        if k in hyp_files
    ]

def load_hypnogram(path):
    annotations = mne.read_annotations(path)

    stages = []

    for desc, duration in zip(
        annotations.description,
        annotations.duration
    ):
        if desc in STAGE_MAPPING:
            stages.extend(
                [STAGE_MAPPING[desc]] * int(duration / 30)
            )

    return np.array(stages)


def load_eeg(path, channels=None):
    if channels is None:
        channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick_channels(channels)

    data = raw.get_data()

    return data


def create_epochs(eeg, epoch_length=30):
    samples_per_epoch = int(epoch_length * 100)
    n_epochs = eeg.shape[1] // samples_per_epoch
    eeg = eeg[:, :n_epochs * samples_per_epoch]
    epochs = eeg.reshape(eeg.shape[0], n_epochs, samples_per_epoch)

    return np.transpose(epochs, (1, 0, 2))


def load_data():
    data_dir = r"D:\Datasets\sleep-edfx"
    pairs = get_matched_pairs(data_dir)

    X = []
    y = []

    for psg_path, hyp_path in pairs:
        eeg = load_eeg(psg_path)
        epochs = create_epochs(eeg)
        labels = load_hypnogram(hyp_path)
        n = min(len(epochs), len(labels))
        X.extend(epochs[:n])
        y.extend(labels[:n])

    X = np.array(X)
    y = np.array(y)

    return X, y
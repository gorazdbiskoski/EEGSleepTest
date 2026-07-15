import os
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

    raw = mne.io.read_raw_edf(path,preload=False,verbose=False)
    raw.pick(EEG_CHANNELS)
    raw.load_data()
    raw.resample(TARGET_SFREQ)

    eeg = raw.get_data()

    return eeg



def create_epochs(eeg):

    samples_per_epoch = 30 * TARGET_SFREQ
    n_epochs = eeg.shape[1] // samples_per_epoch
    eeg = eeg[:, :n_epochs * samples_per_epoch]
    epochs = eeg.reshape(eeg.shape[0], n_epochs,samples_per_epoch)

    epochs = np.transpose(epochs,(1, 0, 2))

    return epochs



def normalize_epochs(X):

    mean = X.mean(axis=-1,keepdims=True)
    std = X.std(axis=-1,keepdims=True)

    return (X - mean) / (std + 1e-8)



def load_data():

    data_dir = r"D:\Datasets\Haaglanden Medisch Centrum sleep staging database"
    pairs = get_matched_pairs(data_dir)
    print(f"Found {len(pairs)} PSG/Hypnogram pairs")

    X = []
    y = []

    for i, (psg_path, hyp_path) in enumerate(pairs, start=1):

        print(f"Loading {i}/{len(pairs)}: {os.path.basename(psg_path)}")

        eeg = load_eeg(psg_path)
        epochs = create_epochs(eeg)
        labels = load_hypnogram(hyp_path)

        n = min(len(epochs),len(labels))
        X.extend(epochs[:n])
        y.extend(labels[:n])


    X = np.array(X)
    y = np.array(y)

    X = normalize_epochs(X)

    print("\nFinished loading")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y



X, y = load_data()
print("\nExample:")
print("First epoch shape:", X[0].shape)
print("First label:", y[0])
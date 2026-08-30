import os

from dotenv import load_dotenv

from data.edf_common import (
    SAMPLES_PER_EPOCH,
    STAGE_MAPPING,
    TARGET_SFREQ,
    create_epochs,
    load_dataset,
    load_eeg as _load_eeg,
    load_hypnogram,
    normalize_epochs,
    warn_about_orphans,
)

load_dotenv()

EEG_CHANNELS = [
    "EEG C4-M1",
    "EEG O2-M1",
]

DATA_HAANGLANDEN = os.getenv("DATA_HAANGLANDEN")


def load_eeg(path):
    return _load_eeg(path, EEG_CHANNELS)


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

    warn_about_orphans(psg_files, hyp_files, "Haaglanden")

    return [
        (psg_files[key], hyp_files[key])
        for key in sorted(psg_files)
        if key in hyp_files
    ]


def load_data_haaglanden(n_samples=None, random_state=42):
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

    return load_dataset(
        pairs,
        channels=EEG_CHANNELS,
        dataset_label="Haaglanden",
        n_samples=n_samples,
        random_state=random_state,
    )


if __name__ == "__main__":
    X, y = load_data_haaglanden()
    print("\nExample:")
    print("First epoch shape:", X[0].shape)
    print("First label:", y[0])

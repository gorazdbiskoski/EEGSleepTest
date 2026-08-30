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

EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]

DATA_SLEEP_EDFX = os.getenv("DATA_SLEEP_EDFX")


def load_eeg(path):
    return _load_eeg(path, EEG_CHANNELS)


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

    warn_about_orphans(psg_files, hyp_files, "Sleep-EDF")

    return [(psg_files[key], hyp_files[key]) for key in sorted(psg_files) if key in hyp_files]


def load_data_sleep_edfx(n_samples=None, random_state=42):
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

    return load_dataset(
        pairs,
        channels=EEG_CHANNELS,
        dataset_label="Sleep-EDF",
        n_samples=n_samples,
        random_state=random_state,
    )


if __name__ == "__main__":
    X, y = load_data_sleep_edfx()
    print("\nExample:")
    print("First epoch shape:", X[0].shape)
    print("First label:", y[0])

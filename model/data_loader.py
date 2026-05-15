import mne
import numpy as np
import pandas as pd
import os

STAGE_MAPPING = {
    "Sleep stage W":  "W",
    "Sleep stage 1":  "N1",
    "Sleep stage 2":  "N2",
    "Sleep stage 3":  "N3",
    "Sleep stage 4":  "N3",
    "Sleep stage R":  "REM",
}


def get_matched_pairs(data_dir):
    """
    Match PSG files with their corresponding Hypnogram files.
    Sleep-EDF naming convention:
        PSG:       SC4001E0-PSG.edf
        Hypnogram: SC4001EC-Hypnogram.edf
    The first 6 characters are the shared subject/session key.
    """
    psg_files = {}
    hyp_files = {}

    for f in os.listdir(data_dir):
        if f.endswith("-PSG.edf"):
            key = f[:6]
            psg_files[key] = os.path.join(data_dir, f)
        elif f.endswith("-Hypnogram.edf"):
            key = f[:6]
            hyp_files[key] = os.path.join(data_dir, f)

    pairs = []
    for key in psg_files:
        if key in hyp_files:
            pairs.append((psg_files[key], hyp_files[key]))
        else:
            print(f"  [warn] No hypnogram found for {psg_files[key]}, skipping.")

    print(f"Found {len(pairs)} matched PSG/Hypnogram pairs.")
    return pairs


def load_hypnogram(hypnogram_path):
    """
    Load a hypnogram EDF and return a DataFrame with
    columns: Onset, Duration, Stage (mapped label).
    Rows whose description is not in STAGE_MAPPING are dropped.
    """
    annotations = mne.read_annotations(hypnogram_path)

    rows = []
    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        stage = STAGE_MAPPING.get(description)
        if stage is not None:
            rows.append({"Onset": onset, "Duration": duration, "Stage": stage})

    return pd.DataFrame(rows)


def compute_features(hypnogram_df):
    """
    Compute 6 sleep-quality features from a single night's hypnogram.
    Returns a dict (one row of features + the real target).
    """
    TST = hypnogram_df[hypnogram_df["Stage"] != "W"]["Duration"].sum() / 60

    N3_total = hypnogram_df[hypnogram_df["Stage"] == "N3"]["Duration"].sum()
    N3_percentage = (N3_total / (TST * 60)) * 100 if TST > 0 else 0.0

    REM_total  = hypnogram_df[hypnogram_df["Stage"] == "REM"]["Duration"].sum()
    REM_percentage = (REM_total / (TST * 60)) * 100 if TST > 0 else 0.0

    awakenings = len(hypnogram_df[hypnogram_df["Stage"] == "W"])

    non_wake = hypnogram_df[hypnogram_df["Stage"] != "W"]
    SOL = non_wake["Onset"].min() / 60 if not non_wake.empty else 0.0

    time_in_bed  = hypnogram_df["Duration"].sum() / 60
    sleep_efficiency = (TST / time_in_bed) * 100 if time_in_bed > 0 else 0.0

    return {
        "TST":              TST,
        "N3_percentage":    N3_percentage,
        "REM_percentage":   REM_percentage,
        "Awakenings":       awakenings,
        "SOL":              SOL,
        "Sleep_Efficiency": sleep_efficiency,
        "_target":          sleep_efficiency,
    }


def load_data():
    """
    Main entry point.

    Returns
    -------
    X : np.ndarray, shape (n_subjects, 6, 1)
        Six hypnogram-derived features per night, reshaped for Conv1D input.
    y : np.ndarray, shape (n_subjects,)
        Sleep efficiency score (real target, NOT random noise).
    """
    data_dir = r"C:\sleep-edfx\sleep-cassette"

    pairs = get_matched_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No matched PSG/Hypnogram pairs found in: {data_dir}\n"
            "Check that the path is correct and files follow the SC4xxxxx naming convention."
        )

    features_list = []

    for psg_path, hyp_path in pairs:
        subject = os.path.basename(psg_path)
        print(f"Processing {subject} …")

        try:
            hypnogram_df = load_hypnogram(hyp_path)

            if hypnogram_df.empty:
                print(f"[warn] Empty hypnogram for {subject}, skipping.")
                continue

            row = compute_features(hypnogram_df)
            features_list.append(row)

        except Exception as e:
            print(f"[error] Failed to process {subject}: {e}")
            continue

    if not features_list:
        raise RuntimeError("No subjects were successfully processed.")

    df = pd.DataFrame(features_list)

    feature_cols = ["TST", "N3_percentage", "REM_percentage",
                    "Awakenings", "SOL", "Sleep_Efficiency"]

    X = np.expand_dims(df[feature_cols].values, axis=-1)
    y = df["_target"].values

    print(f"\nLoaded {len(y)} subjects.")
    print(f"X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"Sleep efficiency — mean: {y.mean():.1f}%  "
          f"min: {y.min():.1f}%  max: {y.max():.1f}%")

    return X, y
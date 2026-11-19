import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter

from dotenv import load_dotenv


# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()


# =====================================================
# CONFIG
# =====================================================

SEQUENCE_LENGTH = int(
    os.getenv("SEQUENCE_LENGTH", 10)
)

LABEL_MAP = {
    0: "Stable",
    1: "Medium Risk",
    2: "High Risk",
    3: "Critical"
}


# =====================================================
# LOAD DATA
# =====================================================

def load_data(file_path):

    # -------------------------
    # LOAD CSV
    # -------------------------

    df = pd.read_csv(file_path)

    # -------------------------
    # REQUIRED COLUMNS CHECK
    # -------------------------

    required_columns = [
        "patient_id",
        "hadm_id",
        "time_step",
        "hr",
        "spo2",
        "bp",
        "rr",
        "temp",
        "ecg",
        "label"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns in {file_path}: "
            f"{missing_columns}"
        )

    # -------------------------
    # FEATURES
    # -------------------------

    features = [
        "hr",
        "spo2",
        "bp",
        "rr",
        "temp",
        "ecg"
    ]

    # -------------------------
    # LABEL CLEANING
    # -------------------------

    df["label"] = df["label"].astype(int)

    unique_labels = sorted(
        df["label"].unique().tolist()
    )

    invalid_labels = [
        label
        for label in unique_labels
        if label not in [0, 1, 2, 3]
    ]

    if invalid_labels:

        raise ValueError(
            f"Invalid labels found in {file_path}: "
            f"{invalid_labels}. "
            f"Expected labels: 0, 1, 2, 3"
        )

    # -------------------------
    # SORT DATA
    # -------------------------

    df = df.sort_values(
        by=[
            "patient_id",
            "hadm_id",
            "time_step"
        ]
    ).reset_index(drop=True)

    # -------------------------
    # NORMALIZATION
    # -------------------------

    scaler = MinMaxScaler()

    df[features] = scaler.fit_transform(
        df[features]
    )

    X = []
    y = []

    # -------------------------
    # BUILD TEMPORAL SEQUENCES
    # -------------------------
    # Group by both patient_id and hadm_id so that two admissions
    # from the same patient are not mixed together.
    # Target label is the label at the next time step after the sequence.
    # Example:
    # sequence = time steps 0-9
    # target   = label at time step 10
    # -------------------------

    grouped = df.groupby(
        [
            "patient_id",
            "hadm_id"
        ]
    )

    for _, group_df in grouped:

        group_df = group_df.sort_values(
            by="time_step"
        )

        data = group_df[features].values

        labels = group_df["label"].values.astype(
            int
        )

        if len(data) <= SEQUENCE_LENGTH:

            continue

        for i in range(
            len(data) - SEQUENCE_LENGTH
        ):

            sequence = data[
                i:i + SEQUENCE_LENGTH
            ]

            target = int(
                labels[
                    i + SEQUENCE_LENGTH
                ]
            )

            X.append(sequence)
            y.append(target)

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    if len(X) == 0:

        raise ValueError(
            f"No sequences created from {file_path}. "
            f"Check SEQUENCE_LENGTH={SEQUENCE_LENGTH} "
            f"and available time steps per patient/admission."
        )

    # -------------------------
    # TRAIN TEST SPLIT
    # -------------------------

    label_counts = Counter(y)

    can_stratify = all(
        count >= 2
        for count in label_counts.values()
    ) and len(label_counts) > 1

    stratify_labels = (
        y
        if can_stratify
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels
    )

    # -------------------------
    # PRINT DATASET INFO
    # -------------------------

    print("\n========== DATASET INFO ==========")

    print("CSV Rows:", len(df))
    print("Sequence Length:", SEQUENCE_LENGTH)
    print("Total Sequences:", len(X))

    print("Train Samples:", len(X_train))
    print("Test Samples :", len(X_test))

    print("\nClass Distribution:")

    total = len(y)

    for class_id in [0, 1, 2, 3]:

        count = int(
            np.sum(y == class_id)
        )

        percent = (
            count / total
        ) * 100

        print(
            f"{class_id} - {LABEL_MAP[class_id]}: "
            f"{count} "
            f"({percent:.2f}%)"
        )

    print("==================================\n")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )
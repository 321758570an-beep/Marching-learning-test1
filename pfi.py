"""Five-fold slice PFI for the SVM beat hold-out protocol.

Place this script in the MachineLearningAI directory. It loads all per-record
277-column CSV files directly inside that directory or its mitbih/ folder,
excluding aggregated and train/test CSV files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample


CLASS_IDS = list(range(1, 9))
N_SPLITS = 5
N_SLICES = 11
SAMPLES_PER_SLICE = 25


def balance_fold(X, y, random_state=42):
    """Balance only a fold's fitting data to the mean count of classes 2--8."""
    counts = {class_id: int(np.sum(y == class_id)) for class_id in CLASS_IDS}
    missing = [class_id for class_id, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"The fitting fold has no examples for classes {missing}. "
            "Check the input labels and class counts before running 5-fold PFI."
        )

    target_count = int(np.mean([counts[class_id] for class_id in range(2, 9)]))
    if target_count < 1:
        raise ValueError("The abnormal-class mean is below one sample.")

    X_parts = []
    y_parts = []
    for class_id in CLASS_IDS:
        X_class = X[y == class_id]
        X_sampled = resample(
            X_class,
            replace=(len(X_class) < target_count),
            n_samples=target_count,
            random_state=random_state,
        )
        X_parts.append(X_sampled)
        y_parts.append(np.full(target_count, class_id, dtype=int))

    X_balanced = np.vstack(X_parts)
    y_balanced = np.concatenate(y_parts)
    order = np.random.default_rng(random_state).permutation(len(y_balanced))
    return X_balanced[order], y_balanced[order]


def make_svm():
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        ),
    )


def macro_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, labels=CLASS_IDS,
            average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, labels=CLASS_IDS,
            average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, labels=CLASS_IDS,
            average="macro", zero_division=0
        ),
    }


def main():
    base_dir = Path(__file__).resolve().parent
    excluded_names = {
        "all_data.csv",
        "train_beats.csv",
        "test_beats.csv",
        "train_patients.csv",
        "test_patients.csv",
    }

    # The preprocessing notebook may save record CSVs in the project root
    # or in mitbih/. Do not recursively load train/ and test/ split files.
    source_dirs = [base_dir, base_dir / "mitbih"]
    csv_files = []
    for source_dir in source_dirs:
        if source_dir.is_dir():
            csv_files.extend(source_dir.glob("*.csv"))

    csv_files = sorted({path.resolve() for path in csv_files})
    csv_files = [
        path for path in csv_files
        if path.name.lower() not in excluded_names
    ]
    if not csv_files:
        raise FileNotFoundError(
            f"No per-record CSV files found directly in {base_dir} "
            f"or {base_dir / 'mitbih'}."
        )

    arrays = []
    for csv_path in csv_files:
        arr = np.loadtxt(csv_path, delimiter=",")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 277:
            raise ValueError(
                f"Expected 277 columns (275 waveform + label + patient ID) "
                f"in {csv_path}, but found {arr.shape[1]}. "
                "Check that the source folders contain only per-record "
                "preprocessed beat CSVs."
            )
        arrays.append(arr)

    data = np.vstack(arrays)
    labels_in_file = data[:, -2].astype(int)
    # Per-record files may still contain undetermined label 0; the coursework
    # uses only the eight labelled heartbeat classes.
    keep = np.isin(labels_in_file, CLASS_IDS)
    data = data[keep]
    if len(data) == 0:
        raise ValueError("No labelled classes 1--8 were found in the CSV files.")

    print(f"Loaded {len(csv_files)} per-record CSV files.")
    print(f"Combined labelled data shape: {data.shape}")

    # Match the 75:25 Beat Hold-out split used in data_split_resample.ipynb.
    raw_train, _ = train_test_split(
        data,
        test_size=0.25,
        random_state=48,
    )
    X_raw_train = raw_train[:, :275]
    y_raw_train = raw_train[:, -2].astype(int)

    counts = pd.Series(y_raw_train).value_counts().reindex(CLASS_IDS, fill_value=0)
    print("Original beat-training partition class counts:")
    print(counts.to_string())
    if int(counts.min()) < N_SPLITS:
        raise ValueError(
            "Five-fold stratification requires at least 5 original training "
            "examples in every class. The counts above show a smaller class. "
            "Check that the correct all_data.csv was loaded; do not silently "
            "drop a required heartbeat class."
        )

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42,
    )
    fold_rows = []

    for fold_number, (fit_idx, val_idx) in enumerate(
        skf.split(X_raw_train, y_raw_train),
        start=1,
    ):
        print(f"Fitting fold {fold_number}/{N_SPLITS} ...", flush=True)
        X_fit = X_raw_train[fit_idx]
        y_fit = y_raw_train[fit_idx]
        X_val = X_raw_train[val_idx]
        y_val = y_raw_train[val_idx]

        # Resample fitting rows only; validation rows remain unresampled.
        X_fit_balanced, y_fit_balanced = balance_fold(
            X_fit,
            y_fit,
            random_state=42 + fold_number,
        )
        fold_model = make_svm()
        fold_model.fit(X_fit_balanced, y_fit_balanced)

        baseline_pred = fold_model.predict(X_val)
        baseline = macro_metrics(y_val, baseline_pred)
        baseline_loss = 2.0 * np.mean(y_val != baseline_pred)

        for slice_index in range(N_SLICES):
            print(
                f"  Fold {fold_number}: slice {slice_index + 1}/{N_SLICES}",
                flush=True,
            )
            start = slice_index * SAMPLES_PER_SLICE
            end = start + SAMPLES_PER_SLICE

            # Coursework seed: 42 + 100k + (s - 1), with k and s 1-based.
            rng = np.random.default_rng(
                42 + 100 * fold_number + slice_index
            )
            X_perturbed = X_val.copy()
            original_slice = X_val[:, start:end]

            # Independently shuffle temporal order within each validation beat.
            random_keys = rng.random(original_slice.shape)
            permutation = np.argsort(random_keys, axis=1)
            X_perturbed[:, start:end] = np.take_along_axis(
                original_slice,
                permutation,
                axis=1,
            )

            perturbed_pred = fold_model.predict(X_perturbed)
            perturbed = macro_metrics(y_val, perturbed_pred)
            perturbed_loss = 2.0 * np.mean(y_val != perturbed_pred)

            fold_rows.append({
                "fold": fold_number,
                "slice": slice_index + 1,
                "start": start,
                "end": end - 1,
                # Increase in hard one-hot squared loss.
                "I_s": perturbed_loss - baseline_loss,
                # Metric drops in percentage points; positive means worse.
                "Delta Acc": 100.0 * (baseline["acc"] - perturbed["acc"]),
                "Delta P": 100.0 * (
                    baseline["precision"] - perturbed["precision"]
                ),
                "Delta R": 100.0 * (
                    baseline["recall"] - perturbed["recall"]
                ),
                "Delta F1": 100.0 * (baseline["f1"] - perturbed["f1"]),
            })

    fold_df = pd.DataFrame(fold_rows)
    required_columns = {
        "fold", "slice", "start", "end", "I_s",
        "Delta Acc", "Delta P", "Delta R", "Delta F1",
    }
    missing_columns = required_columns - set(fold_df.columns)
    if fold_df.empty or missing_columns:
        raise RuntimeError(
            "PFI result rows were not created. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    summary = (
        fold_df
        .groupby(["slice", "start", "end"], as_index=False)
        [["I_s", "Delta Acc", "Delta P", "Delta R", "Delta F1"]]
        .mean()
    )
    summary = summary.rename(columns={
        "slice": "Slice",
        "start": "Indices start",
        "end": "Indices end",
        "Delta Acc": r"$\Delta$Acc.",
        "Delta P": r"$\Delta$P",
        "Delta R": r"$\Delta$R",
        "Delta F1": r"$\Delta$F1",
    })
    summary["Indices"] = (
        summary["Indices start"].astype(int).astype(str)
        + "--"
        + summary["Indices end"].astype(int).astype(str)
    )
    summary = summary[[
        "Slice", "Indices", "I_s",
        r"$\Delta$Acc.", r"$\Delta$P", r"$\Delta$R", r"$\Delta$F1",
    ]]

    print("\nMean slice PFI across folds:")
    display_df = summary.copy()
    for column in display_df.columns[2:]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.5f}")
    print(display_df.to_string(index=False))

    print("\nLaTeX table:")
    print(summary.to_latex(index=False, escape=False, float_format="%.5f"))


if __name__ == "__main__":
    main()

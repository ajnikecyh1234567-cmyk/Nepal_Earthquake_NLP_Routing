import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



INPUT_FILE = "messages_with_priority.csv"
OUTPUT_DIR = Path("baseline_results")

TEST_SIZE = 0.20
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




LABEL_ORDER = [
    "injured_or_dead_people",
    "missing_trapped_or_found_people",
    "displaced_people_and_evacuations",
    "infrastructure_and_utilities_damage",
    "caution_and_advice",
    "donation_needs_or_offers_or_volunteering_services",
    "other_useful_information",
    "sympathy_and_emotional_support",
    "not_related_or_irrelevant",
]

EXPECTED_IRRELEVANT_LABEL = "not_related_or_irrelevant"

PRIORITY_MAP = {
    "injured_or_dead_people": "high",
    "missing_trapped_or_found_people": "high",

    "displaced_people_and_evacuations": "medium_high",
    "infrastructure_and_utilities_damage": "medium_high",
    "caution_and_advice": "medium_high",

    "donation_needs_or_offers_or_volunteering_services": "medium",
    "other_useful_information": "medium",

    "sympathy_and_emotional_support": "low",
    "not_related_or_irrelevant": "low",
}

PRIORITY_SCORE_MAP = {
    "high": 1,
    "medium_high": 2,
    "medium": 3,
    "low": 4,
}

PRIORITY_ORDER = [1, 2, 3, 4]
PRIORITY_NAMES = ["High", "Medium-high", "Medium", "Low"]



df = pd.read_csv(
    INPUT_FILE,
    dtype={"tweet_id": "string"},
)

required_columns = {
    "tweet_id",
    "tweet_text",
    "clean_text",
    "label",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if df["label"].isna().any():
    raise ValueError("The label column contains missing values.")

df["tweet_id"] = df["tweet_id"].astype("string")
df["clean_text"] = df["clean_text"].fillna("").astype(str)

empty_text_count = df["clean_text"].str.strip().eq("").sum()
if empty_text_count > 0:
    print(
        f"Warning: {empty_text_count} rows have empty clean_text values."
    )

duplicate_id_count = df["tweet_id"].duplicated().sum()
if duplicate_id_count > 0:
    print(
        f"Warning: {duplicate_id_count} duplicated tweet IDs were found."
    )

unknown_labels = set(df["label"].unique()) - set(PRIORITY_MAP)

if unknown_labels:
    raise ValueError(
        "The following labels are missing from PRIORITY_MAP: "
        f"{sorted(unknown_labels)}"
    )

label_order = [
    label for label in [
        "injured_or_dead_people",
        "missing_trapped_or_found_people",
        "displaced_people_and_evacuations",
        "infrastructure_and_utilities_damage",
        "caution_and_advice",
        "donation_needs_or_offers_or_volunteering_services",
        "other_useful_information",
        "sympathy_and_emotional_support",
        "not_related_or_irrelevant",
    ]
    if label in set(df["label"].unique())
]

print("Data loaded successfully.")
print("Dataset shape:", df.shape)
print("Number of labels:", df["label"].nunique())



all_indices = df.index.to_numpy()

train_idx, test_idx = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

train_df.to_csv(
    OUTPUT_DIR / "train_split.csv",
    index=False,
)

test_df.to_csv(
    OUTPUT_DIR / "test_split.csv",
    index=False,
)

split_series = pd.Series("train", index=df.index)
split_series.loc[test_idx] = "test"

split_manifest = df[["tweet_id", "label"]].copy()
split_manifest["split"] = split_series

split_manifest.to_csv(
    OUTPUT_DIR / "split_manifest.csv",
    index=False,
)

print("Training data size:", len(train_df))
print("Testing data size:", len(test_df))




model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        ),
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])




X_train = train_df["clean_text"]
y_train = train_df["label"]

X_test = test_df["clean_text"]
y_test = test_df["label"]

model.fit(X_train, y_train)

pred_label = model.predict(X_test)
pred_proba = model.predict_proba(X_test)

class_names = model.named_steps["clf"].classes_
class_to_index = {
    label: index
    for index, label in enumerate(class_names)
}

prediction_confidence = pred_proba.max(axis=1)

true_label_confidence = np.array([
    pred_proba[row_index, class_to_index[true_label]]
    for row_index, true_label in enumerate(y_test)
])

print("Model training completed.")




label_accuracy = accuracy_score(y_test, pred_label)
label_macro_f1 = f1_score(
    y_test,
    pred_label,
    average="macro",
)
label_weighted_f1 = f1_score(
    y_test,
    pred_label,
    average="weighted",
)

print("\n==============================")
print("Label Classification Results")
print("==============================")
print("Accuracy:", label_accuracy)
print("Macro-F1:", label_macro_f1)
print("Weighted-F1:", label_weighted_f1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred_label,
        labels=label_order,
        zero_division=0,
    )
)

label_report = classification_report(
    y_test,
    pred_label,
    labels=label_order,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(label_report).T.to_csv(
    OUTPUT_DIR / "label_classification_report.csv"
)




results_df = test_df.copy()

results_df["true_label"] = y_test.to_numpy()
results_df["predicted_label"] = pred_label
results_df["prediction_confidence"] = prediction_confidence
results_df["true_label_confidence"] = true_label_confidence

results_df["label_correct"] = (
    results_df["true_label"]
    == results_df["predicted_label"]
)


results_df["true_priority_level"] = (
    results_df["true_label"].map(PRIORITY_MAP)
)

results_df["true_priority_score"] = (
    results_df["true_priority_level"].map(PRIORITY_SCORE_MAP)
)

results_df["predicted_priority_level"] = (
    results_df["predicted_label"].map(PRIORITY_MAP)
)

results_df["predicted_priority_score"] = (
    results_df["predicted_priority_level"].map(PRIORITY_SCORE_MAP)
)

results_df["priority_correct"] = (
    results_df["true_priority_score"]
    == results_df["predicted_priority_score"]
)




priority_accuracy = accuracy_score(
    results_df["true_priority_score"],
    results_df["predicted_priority_score"],
)

priority_macro_f1 = f1_score(
    results_df["true_priority_score"],
    results_df["predicted_priority_score"],
    average="macro",
)

priority_weighted_f1 = f1_score(
    results_df["true_priority_score"],
    results_df["predicted_priority_score"],
    average="weighted",
)

print("\n==============================")
print("Priority Classification Results")
print("==============================")
print("Accuracy:", priority_accuracy)
print("Macro-F1:", priority_macro_f1)
print("Weighted-F1:", priority_weighted_f1)

print("\nPriority Classification Report:")
print(
    classification_report(
        results_df["true_priority_score"],
        results_df["predicted_priority_score"],
        labels=PRIORITY_ORDER,
        target_names=PRIORITY_NAMES,
        zero_division=0,
    )
)

priority_report = classification_report(
    results_df["true_priority_score"],
    results_df["predicted_priority_score"],
    labels=PRIORITY_ORDER,
    target_names=PRIORITY_NAMES,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(priority_report).T.to_csv(
    OUTPUT_DIR / "priority_classification_report.csv"
)




results_df["priority_signed_error"] = (
    results_df["predicted_priority_score"]
    - results_df["true_priority_score"]
)

results_df["priority_distance"] = (
    results_df["priority_signed_error"].abs()
)


results_df["under_prioritised"] = (
    results_df["priority_signed_error"] > 0
)

results_df["over_prioritised"] = (
    results_df["priority_signed_error"] < 0
)

results_df["severe_priority_error"] = (
    results_df["priority_distance"] >= 2
)

high_mask = results_df["true_priority_score"] == 1

high_priority_recall = (
    results_df.loc[
        high_mask,
        "predicted_priority_score",
    ].eq(1).mean()
)

under_priority_rate = results_df["under_prioritised"].mean()
over_priority_rate = results_df["over_prioritised"].mean()
severe_priority_error_rate = (
    results_df["severe_priority_error"].mean()
)

high_to_low_mask = (
    (results_df["true_priority_score"] == 1)
    & (results_df["predicted_priority_score"] == 4)
)

high_to_low_count = int(high_to_low_mask.sum())
high_to_low_rate_among_high = (
    high_to_low_count / int(high_mask.sum())
    if int(high_mask.sum()) > 0
    else np.nan
)

print("\n==============================")
print("Emergency Priority Error Analysis")
print("==============================")
print("Under-prioritisation rate:", under_priority_rate)
print("Over-prioritisation rate:", over_priority_rate)
print(
    "Severe priority error rate:",
    severe_priority_error_rate,
)
print("High-priority recall:", high_priority_recall)
print("High-to-Low error count:", high_to_low_count)
print(
    "High-to-Low rate among true High messages:",
    high_to_low_rate_among_high,
)

metrics_summary = pd.DataFrame([
    {
        "metric": "label_accuracy",
        "value": label_accuracy,
    },
    {
        "metric": "label_macro_f1",
        "value": label_macro_f1,
    },
    {
        "metric": "label_weighted_f1",
        "value": label_weighted_f1,
    },
    {
        "metric": "priority_accuracy",
        "value": priority_accuracy,
    },
    {
        "metric": "priority_macro_f1",
        "value": priority_macro_f1,
    },
    {
        "metric": "priority_weighted_f1",
        "value": priority_weighted_f1,
    },
    {
        "metric": "under_prioritisation_rate",
        "value": under_priority_rate,
    },
    {
        "metric": "over_prioritisation_rate",
        "value": over_priority_rate,
    },
    {
        "metric": "severe_priority_error_rate",
        "value": severe_priority_error_rate,
    },
    {
        "metric": "high_priority_recall",
        "value": high_priority_recall,
    },
    {
        "metric": "high_to_low_count",
        "value": high_to_low_count,
    },
    {
        "metric": "high_to_low_rate_among_high",
        "value": high_to_low_rate_among_high,
    },
])

metrics_summary.to_csv(
    OUTPUT_DIR / "metrics_summary.csv",
    index=False,
)




results_df.to_csv(
    OUTPUT_DIR / "classification_results_with_confidence.csv",
    index=False,
)

priority_errors = results_df[
    ~results_df["priority_correct"]
].copy()

priority_errors = priority_errors.sort_values(
    by=[
        "priority_distance",
        "prediction_confidence",
    ],
    ascending=[False, False],
)

priority_errors.to_csv(
    OUTPUT_DIR / "all_priority_errors.csv",
    index=False,
)

high_priority_errors = results_df[
    high_mask
    & (
        results_df["predicted_priority_score"] != 1
    )
].copy()

high_priority_errors = high_priority_errors.sort_values(
    by=[
        "priority_signed_error",
        "prediction_confidence",
    ],
    ascending=[False, False],
)

high_priority_errors.to_csv(
    OUTPUT_DIR / "high_priority_errors.csv",
    index=False,
)

results_df[high_to_low_mask].to_csv(
    OUTPUT_DIR / "high_to_low_errors.csv",
    index=False,
)




def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    display_labels,
    title,
    output_path,
    normalize=None,
    values_format=None,
    figure_size=(12, 10),
):
    fig, ax = plt.subplots(figsize=figure_size)

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=display_labels,
        normalize=normalize,
        values_format=values_format,
        xticks_rotation=90,
        ax=ax,
    )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


save_confusion_matrix(
    y_true=y_test,
    y_pred=pred_label,
    labels=label_order,
    display_labels=label_order,
    title=(
        "TF-IDF Logistic Regression "
        "Confusion Matrix (Counts)"
    ),
    output_path=(
        OUTPUT_DIR
        / "label_confusion_matrix_counts.png"
    ),
    normalize=None,
    values_format="d",
    figure_size=(15, 13),
)

save_confusion_matrix(
    y_true=y_test,
    y_pred=pred_label,
    labels=label_order,
    display_labels=label_order,
    title=(
        "TF-IDF Logistic Regression "
        "Confusion Matrix (Normalised)"
    ),
    output_path=(
        OUTPUT_DIR
        / "label_confusion_matrix_normalised.png"
    ),
    normalize="true",
    values_format=".2f",
    figure_size=(15, 13),
)

save_confusion_matrix(
    y_true=results_df["true_priority_score"],
    y_pred=results_df["predicted_priority_score"],
    labels=PRIORITY_ORDER,
    display_labels=PRIORITY_NAMES,
    title="TF-IDF Priority Confusion Matrix (Counts)",
    output_path=(
        OUTPUT_DIR
        / "priority_confusion_matrix_counts.png"
    ),
    normalize=None,
    values_format="d",
    figure_size=(8, 7),
)

save_confusion_matrix(
    y_true=results_df["true_priority_score"],
    y_pred=results_df["predicted_priority_score"],
    labels=PRIORITY_ORDER,
    display_labels=PRIORITY_NAMES,
    title=(
        "TF-IDF Priority Confusion Matrix "
        "(Normalised)"
    ),
    output_path=(
        OUTPUT_DIR
        / "priority_confusion_matrix_normalised.png"
    ),
    normalize="true",
    values_format=".2f",
    figure_size=(8, 7),
)




joblib.dump(
    model,
    OUTPUT_DIR
    / "tfidf_logistic_regression_model.joblib",
)

print("\nAll files were saved to:")
print(OUTPUT_DIR.resolve())

print("\nImportant files for the next BERT stage:")
print(OUTPUT_DIR / "split_manifest.csv")
print(OUTPUT_DIR / "train_split.csv")
print(OUTPUT_DIR / "test_split.csv")

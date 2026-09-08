import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)




MODEL_NAME = "google-bert/bert-base-uncased"

TRAIN_FILE_CANDIDATES = [
    Path("baseline_results/train_split.csv"),
    Path("train_split.csv"),
]

TEST_FILE_CANDIDATES = [
    Path("baseline_results/test_split.csv"),
    Path("test_split.csv"),
]

BASELINE_METRICS_CANDIDATES = [
    Path("baseline_results/metrics_summary.csv"),
    Path("metrics_summary.csv"),
]

OUTPUT_DIR = Path("bert_results")
MODEL_DIR = OUTPUT_DIR / "bert_model"

RANDOM_STATE = 42
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10


USE_CLASS_WEIGHTS = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)




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

LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)



def resolve_existing_file(candidates, description):
    for path in candidates:
        if path.exists():
            return path

    candidate_text = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find {description}. Checked:\n{candidate_text}"
    )


TRAIN_FILE = resolve_existing_file(
    TRAIN_FILE_CANDIDATES,
    "the fixed training split",
)

TEST_FILE = resolve_existing_file(
    TEST_FILE_CANDIDATES,
    "the fixed test split",
)

print("Training file:", TRAIN_FILE.resolve())
print("Test file:", TEST_FILE.resolve())



train_df = pd.read_csv(
    TRAIN_FILE,
    dtype={"tweet_id": "string"},
)

test_df = pd.read_csv(
    TEST_FILE,
    dtype={"tweet_id": "string"},
)

required_columns = {
    "tweet_id",
    "tweet_text",
    "clean_text",
    "label",
}

for split_name, df in [("train", train_df), ("test", test_df)]:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{split_name} split is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["label"].isna().any():
        raise ValueError(
            f"{split_name} split contains missing labels."
        )

    df["tweet_id"] = df["tweet_id"].astype("string")
    df["clean_text"] = df["clean_text"].fillna("").astype(str)

    unknown_labels = set(df["label"].unique()) - set(LABEL_ORDER)

    if unknown_labels:
        raise ValueError(
            f"{split_name} split contains unknown labels: "
            f"{sorted(unknown_labels)}"
        )


print("\nFixed split loaded successfully.")
print("Training data size:", len(train_df))
print("Testing data size:", len(test_df))
print("Number of labels:", train_df["label"].nunique())

print("\nTraining label distribution:")
print(train_df["label"].value_counts())

print("\nTesting label distribution:")
print(test_df["label"].value_counts())



print("\nLoading tokenizer:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class EarthquakeTweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe["clean_text"].tolist()
        self.labels = [
            LABEL_TO_ID[label]
            for label in dataframe["label"].tolist()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        }


train_dataset = EarthquakeTweetDataset(
    train_df,
    tokenizer,
    MAX_LENGTH,
)

test_dataset = EarthquakeTweetDataset(
    test_df,
    tokenizer,
    MAX_LENGTH,
)

train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_STATE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=train_generator,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("\nTraining device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_ORDER),
    label2id=LABEL_TO_ID,
    id2label=ID_TO_LABEL,
)

model.to(device)



train_label_ids = np.array([
    LABEL_TO_ID[label]
    for label in train_df["label"].tolist()
])

if USE_CLASS_WEIGHTS:
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABEL_ORDER)),
        y=train_label_ids,
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    print("\nClass weights:")
    for label, weight in zip(LABEL_ORDER, class_weights):
        print(f"{label}: {weight:.4f}")
else:
    criterion = torch.nn.CrossEntropyLoss()




optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

total_training_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_training_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)




training_history = []

print("\n==============================")
print("BERT Fine-tuning")
print("==============================")

for epoch in range(1, EPOCHS + 1):
    model.train()

    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_index, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if batch_index % 50 == 0 or batch_index == len(train_loader):
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    mean_train_loss = epoch_loss / len(train_loader)
    train_accuracy = correct / total

    training_history.append({
        "epoch": epoch,
        "train_loss": mean_train_loss,
        "train_accuracy": train_accuracy,
    })

    print(
        f"Epoch {epoch} completed | "
        f"Mean loss: {mean_train_loss:.4f} | "
        f"Training accuracy: {train_accuracy:.4f}"
    )


pd.DataFrame(training_history).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False,
)




model.eval()

all_logits = []
all_true_ids = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        all_logits.append(
            outputs.logits.cpu()
        )

        all_true_ids.append(
            batch["labels"].cpu()
        )


logits = torch.cat(all_logits, dim=0)
true_ids = torch.cat(all_true_ids, dim=0).numpy()

probabilities = torch.softmax(
    logits,
    dim=1,
).numpy()

pred_ids = probabilities.argmax(axis=1)

pred_label = np.array([
    ID_TO_LABEL[int(i)]
    for i in pred_ids
])

y_test = test_df["label"].to_numpy()

prediction_confidence = probabilities.max(axis=1)

true_label_confidence = np.array([
    probabilities[row_index, LABEL_TO_ID[true_label]]
    for row_index, true_label in enumerate(y_test)
])




label_accuracy = accuracy_score(
    y_test,
    pred_label,
)

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
print("BERT Label Classification Results")
print("==============================")
print("Accuracy:", label_accuracy)
print("Macro-F1:", label_macro_f1)
print("Weighted-F1:", label_weighted_f1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred_label,
        labels=LABEL_ORDER,
        zero_division=0,
    )
)

label_report = classification_report(
    y_test,
    pred_label,
    labels=LABEL_ORDER,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(label_report).T.to_csv(
    OUTPUT_DIR / "label_classification_report.csv"
)




results_df = test_df.copy()

results_df["true_label"] = y_test
results_df["predicted_label"] = pred_label

results_df["prediction_confidence"] = (
    prediction_confidence
)

results_df["true_label_confidence"] = (
    true_label_confidence
)

results_df["label_correct"] = (
    results_df["true_label"]
    == results_df["predicted_label"]
)

results_df["true_priority_level"] = (
    results_df["true_label"].map(PRIORITY_MAP)
)

results_df["true_priority_score"] = (
    results_df["true_priority_level"].map(
        PRIORITY_SCORE_MAP
    )
)

results_df["predicted_priority_level"] = (
    results_df["predicted_label"].map(PRIORITY_MAP)
)

results_df["predicted_priority_score"] = (
    results_df["predicted_priority_level"].map(
        PRIORITY_SCORE_MAP
    )
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
print("BERT Priority Classification Results")
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

# A larger score means a lower routing priority.
results_df["under_prioritised"] = (
    results_df["priority_signed_error"] > 0
)

results_df["over_prioritised"] = (
    results_df["priority_signed_error"] < 0
)

results_df["severe_priority_error"] = (
    results_df["priority_distance"] >= 2
)

high_mask = (
    results_df["true_priority_score"] == 1
)

high_priority_recall = (
    results_df.loc[
        high_mask,
        "predicted_priority_score",
    ].eq(1).mean()
)

under_priority_rate = (
    results_df["under_prioritised"].mean()
)

over_priority_rate = (
    results_df["over_prioritised"].mean()
)

severe_priority_error_rate = (
    results_df["severe_priority_error"].mean()
)

high_to_low_mask = (
    (results_df["true_priority_score"] == 1)
    & (results_df["predicted_priority_score"] == 4)
)

high_to_low_count = int(
    high_to_low_mask.sum()
)

high_to_low_rate_among_high = (
    high_to_low_count / int(high_mask.sum())
    if int(high_mask.sum()) > 0
    else np.nan
)

print("\n==============================")
print("BERT Emergency Priority Error Analysis")
print("==============================")
print(
    "Under-prioritisation rate:",
    under_priority_rate,
)
print(
    "Over-prioritisation rate:",
    over_priority_rate,
)
print(
    "Severe priority error rate:",
    severe_priority_error_rate,
)
print(
    "High-priority recall:",
    high_priority_recall,
)
print(
    "High-to-Low error count:",
    high_to_low_count,
)
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
    OUTPUT_DIR
    / "classification_results_with_confidence.csv",
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
    fig, ax = plt.subplots(
        figsize=figure_size
    )

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
    labels=LABEL_ORDER,
    display_labels=LABEL_ORDER,
    title="BERT Confusion Matrix (Counts)",
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
    labels=LABEL_ORDER,
    display_labels=LABEL_ORDER,
    title="BERT Confusion Matrix (Normalised)",
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
    title="BERT Priority Confusion Matrix (Counts)",
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
    title="BERT Priority Confusion Matrix (Normalised)",
    output_path=(
        OUTPUT_DIR
        / "priority_confusion_matrix_normalised.png"
    ),
    normalize="true",
    values_format=".2f",
    figure_size=(8, 7),
)




model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

run_config = {
    "model_name": MODEL_NAME,
    "random_state": RANDOM_STATE,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "use_class_weights": USE_CLASS_WEIGHTS,
    "train_file": str(TRAIN_FILE),
    "test_file": str(TEST_FILE),
    "train_size": len(train_df),
    "test_size": len(test_df),
    "device": str(device),
}

with open(
    OUTPUT_DIR / "run_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        run_config,
        f,
        indent=2,
    )




baseline_metrics_file = None

for candidate in BASELINE_METRICS_CANDIDATES:
    if candidate.exists():
        baseline_metrics_file = candidate
        break

if baseline_metrics_file is not None:
    baseline_metrics = pd.read_csv(
        baseline_metrics_file
    )

    comparison = baseline_metrics.merge(
        metrics_summary,
        on="metric",
        how="outer",
        suffixes=("_lr", "_bert"),
    )

    comparison["bert_minus_lr"] = (
        comparison["value_bert"]
        - comparison["value_lr"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "bert_vs_lr_metrics.csv",
        index=False,
    )

    print(
        "\nLR vs BERT metric comparison saved to:",
        OUTPUT_DIR / "bert_vs_lr_metrics.csv",
    )
else:
    print(
        "\nBaseline metrics_summary.csv was not found. "
        "BERT results are still saved normally."
    )


print("\n==============================")
print("BERT stage completed")
print("==============================")
print("All BERT files were saved to:")
print(OUTPUT_DIR.resolve())

print("\nMost important files:")
print(
    OUTPUT_DIR
    / "classification_results_with_confidence.csv"
)
print(
    OUTPUT_DIR
    / "metrics_summary.csv"
)
print(
    OUTPUT_DIR
    / "priority_classification_report.csv"
)
print(
    OUTPUT_DIR
    / "high_priority_errors.csv"
)
print(
    OUTPUT_DIR
    / "priority_confusion_matrix_counts.png"
)
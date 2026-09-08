from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

LR_HIGHLOAD_FILE = BASE_DIR / "nepal_simulation_messages_highload.csv"

BERT_RESULTS_CANDIDATES = [
    BASE_DIR / "bert_results" / "classification_results_with_confidence.csv",
    BASE_DIR / "classification_results_with_confidence.csv",
]

OUTPUT_FILE = BASE_DIR / "nepal_simulation_messages_highload_bert.csv"

EXPECTED_MESSAGE_COUNT = 959
VALID_PRIORITIES = {1, 2, 3, 4}


def resolve_bert_results_file() -> Path:
    for path in BERT_RESULTS_CANDIDATES:
        if path.exists():
            return path

    checked = "\n".join(f"  - {p}" for p in BERT_RESULTS_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the BERT classification results.\n"
        "Checked:\n"
        f"{checked}"
    )


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return dataframe


def load_lr_highload_schedule() -> pd.DataFrame:
    if not LR_HIGHLOAD_FILE.exists():
        raise FileNotFoundError(
            "The existing LR high-load schedule was not found:\n"
            f"{LR_HIGHLOAD_FILE}"
        )

    schedule = pd.read_csv(
        LR_HIGHLOAD_FILE,
        dtype={"tweet_id": "string"},
        encoding="utf-8",
    )
    schedule = clean_column_names(schedule)

    required_columns = {
        "message_id",
        "creation_time",
        "source",
        "destination",
        "size",
        "true_priority",
        "predicted_priority",
        "tweet_id",
    }

    missing = required_columns - set(schedule.columns)
    if missing:
        raise ValueError(
            "The LR high-load schedule is missing columns: "
            f"{sorted(missing)}"
        )

    if schedule["tweet_id"].isna().any():
        raise ValueError("The LR high-load schedule contains missing tweet_id values.")

    schedule["tweet_id"] = schedule["tweet_id"].astype("string")

    return schedule


def load_bert_priority_map() -> tuple[pd.DataFrame, Path]:
    bert_file = resolve_bert_results_file()

    bert = pd.read_csv(
        bert_file,
        dtype={"tweet_id": "string"},
        encoding="utf-8",
    )
    bert = clean_column_names(bert)

    required_columns = {
        "tweet_id",
        "predicted_priority_score",
    }

    missing = required_columns - set(bert.columns)
    if missing:
        raise ValueError(
            "The BERT results file is missing columns: "
            f"{sorted(missing)}"
        )

    bert["tweet_id"] = bert["tweet_id"].astype("string")

    bert["predicted_priority_score"] = pd.to_numeric(
        bert["predicted_priority_score"],
        errors="coerce",
    )

    if bert["predicted_priority_score"].isna().any():
        raise ValueError(
            "BERT predicted_priority_score contains missing "
            "or non-numeric values."
        )

    bert["predicted_priority_score"] = (
        bert["predicted_priority_score"].astype(int)
    )

    if not bert["predicted_priority_score"].isin(VALID_PRIORITIES).all():
        raise ValueError(
            "BERT predicted_priority_score contains values outside 1-4."
        )


    conflicts = (
        bert.groupby("tweet_id")["predicted_priority_score"]
        .nunique()
    )
    conflicting_ids = conflicts[conflicts > 1]

    if not conflicting_ids.empty:
        raise ValueError(
            "Some tweet_id values have conflicting BERT priorities:\n"
            f"{conflicting_ids.head(10)}"
        )

    priority_map = (
        bert[["tweet_id", "predicted_priority_score"]]
        .drop_duplicates(subset=["tweet_id"])
        .set_index("tweet_id")
    )

    return priority_map, bert_file


def build_bert_schedule(
    lr_schedule: pd.DataFrame,
    bert_priority_map: pd.DataFrame,
) -> pd.DataFrame:
    output = lr_schedule.copy()

    bert_priority = (
        output["tweet_id"]
        .map(bert_priority_map["predicted_priority_score"])
    )

    missing_mask = bert_priority.isna()

    if missing_mask.any():
        missing_ids = (
            output.loc[missing_mask, "tweet_id"]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Some tweet IDs in the LR high-load schedule do not have "
            "a BERT prediction.\n"
            f"Example missing IDs: {missing_ids}"
        )

    output["predicted_priority"] = bert_priority.astype(int)

    return output


def validate_fixed_schedule(
    lr_schedule: pd.DataFrame,
    bert_schedule: pd.DataFrame,
) -> None:
    if len(bert_schedule) != len(lr_schedule):
        raise ValueError("Message count changed unexpectedly.")

    if len(bert_schedule) != EXPECTED_MESSAGE_COUNT:
        print(
            f"Warning: expected {EXPECTED_MESSAGE_COUNT} messages, "
            f"but found {len(bert_schedule)}."
        )

    fixed_columns = [
        "message_id",
        "creation_time",
        "source",
        "destination",
        "size",
        "true_priority",
        "tweet_id",
    ]

    for column in fixed_columns:
        if not lr_schedule[column].equals(bert_schedule[column]):
            raise ValueError(
                f"Fixed field changed unexpectedly: {column}"
            )

    if not bert_schedule["predicted_priority"].isin(VALID_PRIORITIES).all():
        raise ValueError(
            "BERT predicted_priority contains values outside 1-4."
        )

    if bert_schedule.isna().any().any():
        raise ValueError(
            "Missing values are present in the generated BERT schedule."
        )


def print_summary(
    lr_schedule: pd.DataFrame,
    bert_schedule: pd.DataFrame,
    bert_file: Path,
) -> None:
    changed = (
        lr_schedule["predicted_priority"]
        != bert_schedule["predicted_priority"]
    )

    print()
    print("=" * 72)
    print("BERT high-load The ONE schedule created successfully")
    print("=" * 72)

    print(f"BERT results source: {bert_file}")
    print(f"Messages: {len(bert_schedule)}")
    print(
        "Rows whose predicted priority changes from LR to BERT: "
        f"{int(changed.sum())}"
    )

    print("\nTrue-priority distribution:")
    print(
        bert_schedule["true_priority"]
        .value_counts()
        .sort_index()
        .rename_axis("priority")
        .to_string()
    )

    print("\nLR predicted-priority distribution:")
    print(
        lr_schedule["predicted_priority"]
        .value_counts()
        .sort_index()
        .rename_axis("priority")
        .to_string()
    )

    print("\nBERT predicted-priority distribution:")
    print(
        bert_schedule["predicted_priority"]
        .value_counts()
        .sort_index()
        .rename_axis("priority")
        .to_string()
    )

    lr_accuracy = (
        lr_schedule["true_priority"]
        == lr_schedule["predicted_priority"]
    ).mean()

    bert_accuracy = (
        bert_schedule["true_priority"]
        == bert_schedule["predicted_priority"]
    ).mean()

    print(
        f"\nLR priority accuracy in 959-message traffic: "
        f"{lr_accuracy:.4f}"
    )
    print(
        f"BERT priority accuracy in 959-message traffic: "
        f"{bert_accuracy:.4f}"
    )

    print("\nFirst five rows where LR and BERT priorities differ:")
    comparison = pd.DataFrame({
        "message_id": bert_schedule.loc[changed, "message_id"],
        "tweet_id": bert_schedule.loc[changed, "tweet_id"],
        "true_priority": bert_schedule.loc[changed, "true_priority"],
        "lr_predicted_priority": (
            lr_schedule.loc[changed, "predicted_priority"]
        ),
        "bert_predicted_priority": (
            bert_schedule.loc[changed, "predicted_priority"]
        ),
    })

    if comparison.empty:
        print("No predicted-priority differences were found.")
    else:
        print(comparison.head().to_string(index=False))


def main() -> None:
    lr_schedule = load_lr_highload_schedule()

    bert_priority_map, bert_file = load_bert_priority_map()

    bert_schedule = build_bert_schedule(
        lr_schedule,
        bert_priority_map,
    )

    validate_fixed_schedule(
        lr_schedule,
        bert_schedule,
    )

    bert_schedule.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print_summary(
        lr_schedule,
        bert_schedule,
        bert_file,
    )

    print()
    print(f"Saved to:\n{OUTPUT_FILE}")
    print(
        "\nValidation passed: all network fields are unchanged "
        "except predicted_priority."
    )


if __name__ == "__main__":
    main()

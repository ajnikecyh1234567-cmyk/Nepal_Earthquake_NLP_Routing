from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "step2_classification_results.csv"
OUTPUT_FILE = BASE_DIR / "nepal_simulation_messages_highload.csv"

RANDOM_SEED = 1
SIMULATION_END_TIME = 43_200

MIN_MESSAGE_INTERVAL = 30
MAX_MESSAGE_INTERVAL = 60

RESIDENT_MIN = 0
RESIDENT_MAX = 59
RESCUER_MIN = 60
RESCUER_MAX = 79


MIN_MESSAGE_SIZE = 100_000
MAX_MESSAGE_SIZE = 500_000


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalise CSV column names."""
    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return dataframe


def load_classification_results() -> pd.DataFrame:
    """Load and validate the fixed NLP test-set predictions."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Place step2_classification_results.csv in the same "
            "directory as this script."
        )

    dataframe = pd.read_csv(
        INPUT_FILE,
        dtype={"tweet_id": "string"},
        encoding="utf-8",
    )

    dataframe = clean_column_names(dataframe)

    required_columns = {
        "tweet_id",
        "priority_score",
        "predicted_priority_score",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    if dataframe.empty:
        raise ValueError("The classification-results file is empty.")

    dataframe["true_priority_for_one"] = pd.to_numeric(
        dataframe["priority_score"],
        errors="coerce",
    )

    dataframe["predicted_priority_for_one"] = pd.to_numeric(
        dataframe["predicted_priority_score"],
        errors="coerce",
    )

    for column in [
        "true_priority_for_one",
        "predicted_priority_for_one",
    ]:
        if dataframe[column].isna().any():
            raise ValueError(
                f"{column} contains missing or non-numeric values."
            )

        dataframe[column] = dataframe[column].astype(int)

        if not dataframe[column].isin([1, 2, 3, 4]).all():
            raise ValueError(
                f"{column} contains values outside 1-4."
            )

    if dataframe["tweet_id"].isna().any():
        raise ValueError("tweet_id contains missing values.")

    print("Classification results loaded.")
    print(f"NLP test messages available: {len(dataframe)}")

    return dataframe


def create_message_times(
    random_generator: np.random.Generator,
) -> list[int]:
    """Generate increasing creation times for 12 hours."""
    creation_times: list[int] = []
    current_time = 0

    while True:
        current_time += int(
            random_generator.integers(
                MIN_MESSAGE_INTERVAL,
                MAX_MESSAGE_INTERVAL + 1,
            )
        )

        if current_time > SIMULATION_END_TIME:
            break

        creation_times.append(current_time)

    return creation_times


def create_repeated_nlp_pool(
    classification_results: pd.DataFrame,
    number_of_messages: int,
) -> pd.DataFrame:
    """
    Create a deterministic repeated/shuffled NLP pool.

    The high-load schedule may require more network messages than
    there are NLP test rows. In that case, the fixed NLP test set is
    repeated in shuffled cycles. This increases traffic load without
    retraining or changing the classifier.
    """
    cycles = []
    cycle_number = 0
    remaining = number_of_messages

    while remaining > 0:
        shuffled = classification_results.sample(
            frac=1.0,
            replace=False,
            random_state=RANDOM_SEED + cycle_number,
        ).reset_index(drop=True)

        take = min(remaining, len(shuffled))
        cycles.append(shuffled.iloc[:take].copy())

        remaining -= take
        cycle_number += 1

    return pd.concat(cycles, ignore_index=True)


def create_simulation_schedule(
    classification_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create the deterministic high-load simulation schedule."""
    random_generator = np.random.default_rng(RANDOM_SEED)

    creation_times = create_message_times(random_generator)
    number_of_messages = len(creation_times)

    sampled = create_repeated_nlp_pool(
        classification_results,
        number_of_messages,
    )

    schedule = pd.DataFrame({
        "message_id": [
            f"HM{index}"
            for index in range(1, number_of_messages + 1)
        ],
        "creation_time": creation_times,
        "source": random_generator.integers(
            RESIDENT_MIN,
            RESIDENT_MAX + 1,
            size=number_of_messages,
        ),
        "destination": random_generator.integers(
            RESCUER_MIN,
            RESCUER_MAX + 1,
            size=number_of_messages,
        ),
        "size": random_generator.integers(
            MIN_MESSAGE_SIZE,
            MAX_MESSAGE_SIZE + 1,
            size=number_of_messages,
        ),
        "true_priority": (
            sampled["true_priority_for_one"]
            .astype(int)
            .to_numpy()
        ),
        "predicted_priority": (
            sampled["predicted_priority_for_one"]
            .astype(int)
            .to_numpy()
        ),
        "tweet_id": (
            sampled["tweet_id"]
            .astype(str)
            .to_numpy()
        ),
    })

    return schedule


def validate_schedule(schedule: pd.DataFrame) -> None:
    """Validate the generated CSV before saving."""
    if schedule.empty:
        raise ValueError("The generated schedule is empty.")

    if not schedule["message_id"].is_unique:
        raise ValueError("message_id values are not unique.")

    if not schedule["creation_time"].is_monotonic_increasing:
        raise ValueError("creation_time values are not increasing.")

    if not schedule["creation_time"].between(
        0,
        SIMULATION_END_TIME,
    ).all():
        raise ValueError("A creation time is outside 0-43200.")

    if not schedule["source"].between(
        RESIDENT_MIN,
        RESIDENT_MAX,
    ).all():
        raise ValueError("A source address is outside 0-59.")

    if not schedule["destination"].between(
        RESCUER_MIN,
        RESCUER_MAX,
    ).all():
        raise ValueError("A destination address is outside 60-79.")

    if not schedule["size"].between(
        MIN_MESSAGE_SIZE,
        MAX_MESSAGE_SIZE,
    ).all():
        raise ValueError(
            "A message size is outside 100000-500000 bytes."
        )

    for column in ["true_priority", "predicted_priority"]:
        if not schedule[column].isin([1, 2, 3, 4]).all():
            raise ValueError(
                f"{column} contains values outside 1-4."
            )

    if schedule.isna().any().any():
        raise ValueError(
            "Missing values were found:\n"
            f"{schedule.isna().sum()}"
        )


def print_summary(schedule: pd.DataFrame) -> None:
    """Print a reproducibility and distribution summary."""
    print()
    print("=" * 68)
    print("High-load The ONE input created successfully")
    print("=" * 68)
    print(f"Messages created: {len(schedule)}")
    print(
        f"Creation-time range: "
        f"{schedule['creation_time'].min()}-"
        f"{schedule['creation_time'].max()} seconds"
    )
    print(
        f"Configured interval: "
        f"{MIN_MESSAGE_INTERVAL}-"
        f"{MAX_MESSAGE_INTERVAL} seconds"
    )

    print("\nFirst five rows:")
    print(schedule.head().to_string(index=False))

    print("\nTrue-priority distribution:")
    print(
        schedule["true_priority"]
        .value_counts()
        .sort_index()
        .rename_axis("priority")
        .to_string()
    )

    print("\nPredicted-priority distribution:")
    print(
        schedule["predicted_priority"]
        .value_counts()
        .sort_index()
        .rename_axis("priority")
        .to_string()
    )

    priority_accuracy = (
        schedule["true_priority"]
        == schedule["predicted_priority"]
    ).mean()

    print(
        "\nPriority accuracy in generated traffic: "
        f"{priority_accuracy:.4f}"
    )


def main() -> None:
    classification_results = load_classification_results()

    schedule = create_simulation_schedule(
        classification_results
    )

    validate_schedule(schedule)

    schedule.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print_summary(schedule)
    print()
    print(f"Saved to:\n{OUTPUT_FILE}")
    print("High-load CSV validation passed.")


if __name__ == "__main__":
    main()

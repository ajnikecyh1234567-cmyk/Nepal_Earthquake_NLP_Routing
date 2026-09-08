from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "step2_classification_results.csv"
OUTPUT_FILE = BASE_DIR / "nepal_simulation_messages.csv"

RANDOM_SEED = 1

SIMULATION_END_TIME = 43_200

MIN_MESSAGE_INTERVAL = 60
MAX_MESSAGE_INTERVAL = 120

RESIDENT_MIN = 0
RESIDENT_MAX = 59

RESCUER_MIN = 60
RESCUER_MAX = 79

MIN_MESSAGE_SIZE = 100_000
MAX_MESSAGE_SIZE = 500_000


def clean_column_names(dataframe):

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return dataframe


def load_classification_results():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Place step2_classification_results.csv in the same "
            "folder as this script."
        )

    dataframe = pd.read_csv(
        INPUT_FILE,
        dtype={"tweet_id": "string"},
        encoding="utf-8",
    )

    dataframe = clean_column_names(dataframe)

    required_columns = {
        "tweet_id",
        "true_label",
        "predicted_label",
        "priority_score",
        "predicted_priority_score",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            f"{sorted(missing_columns)}\n\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "step2_classification_results.csv contains no rows."
        )

    dataframe["true_priority_for_one"] = pd.to_numeric(
        dataframe["priority_score"],
        errors="coerce",
    )

    dataframe["predicted_priority_for_one"] = pd.to_numeric(
        dataframe["predicted_priority_score"],
        errors="coerce",
    )

    invalid_true_numeric = dataframe.loc[
        dataframe["true_priority_for_one"].isna(),
        ["tweet_id", "priority_score"],
    ]

    if not invalid_true_numeric.empty:
        raise ValueError(
            "Invalid or missing values found in priority_score:\n"
            f"{invalid_true_numeric.head(10)}"
        )

    invalid_predicted_numeric = dataframe.loc[
        dataframe["predicted_priority_for_one"].isna(),
        ["tweet_id", "predicted_priority_score"],
    ]

    if not invalid_predicted_numeric.empty:
        raise ValueError(
            "Invalid or missing values found in "
            "predicted_priority_score:\n"
            f"{invalid_predicted_numeric.head(10)}"
        )

    dataframe["true_priority_for_one"] = (
        dataframe["true_priority_for_one"].astype(int)
    )

    dataframe["predicted_priority_for_one"] = (
        dataframe["predicted_priority_for_one"].astype(int)
    )

    valid_priorities = {1, 2, 3, 4}

    invalid_true_range = dataframe.loc[
        ~dataframe["true_priority_for_one"].isin(valid_priorities),
        ["tweet_id", "priority_score"],
    ]

    if not invalid_true_range.empty:
        raise ValueError(
            "priority_score contains values outside 1-4:\n"
            f"{invalid_true_range.head(10)}"
        )

    invalid_predicted_range = dataframe.loc[
        ~dataframe["predicted_priority_for_one"].isin(
            valid_priorities
        ),
        ["tweet_id", "predicted_priority_score"],
    ]

    if not invalid_predicted_range.empty:
        raise ValueError(
            "predicted_priority_score contains values "
            "outside 1-4:\n"
            f"{invalid_predicted_range.head(10)}"
        )

    if dataframe["tweet_id"].isna().any():
        raise ValueError("tweet_id contains missing values.")

    print("Classification results loaded successfully.")
    print(f"Number of NLP test messages: {len(dataframe)}")

    return dataframe


def create_message_times(random_generator):
    creation_times = []
    current_time = 0

    while True:
        interval = int(
            random_generator.integers(
                MIN_MESSAGE_INTERVAL,
                MAX_MESSAGE_INTERVAL + 1,
            )
        )

        current_time += interval

        if current_time > SIMULATION_END_TIME:
            break

        creation_times.append(current_time)

    return creation_times


def create_simulation_schedule(classification_results):
    random_generator = np.random.default_rng(RANDOM_SEED)

    creation_times = create_message_times(random_generator)
    number_of_messages = len(creation_times)

    use_replacement = (
        len(classification_results) < number_of_messages
    )

    sampled = classification_results.sample(
        n=number_of_messages,
        replace=use_replacement,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    schedule = pd.DataFrame({

        "message_id": [
            f"M{index}"
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


def validate_schedule(schedule):

    if schedule.empty:
        raise ValueError("The generated schedule is empty.")

    if not schedule["message_id"].is_unique:
        raise ValueError("message_id contains duplicates.")

    if not schedule["creation_time"].is_monotonic_increasing:
        raise ValueError(
            "creation_time is not monotonically increasing."
        )

    if not schedule["creation_time"].between(
        0,
        SIMULATION_END_TIME,
    ).all():
        raise ValueError(
            "A creation_time value is outside 0-43200."
        )

    if not schedule["source"].between(
        RESIDENT_MIN,
        RESIDENT_MAX,
    ).all():
        raise ValueError(
            "A source address is outside 0-59."
        )

    if not schedule["destination"].between(
        RESCUER_MIN,
        RESCUER_MAX,
    ).all():
        raise ValueError(
            "A destination address is outside 60-79."
        )

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
        missing_counts = schedule.isna().sum()

        raise ValueError(
            "Missing values were found in the schedule:\n"
            f"{missing_counts}"
        )


def print_summary(schedule):

    print()
    print("=" * 64)
    print("The ONE simulation input created successfully")
    print("=" * 64)

    print(f"Number of messages: {len(schedule)}")
    print(
        f"First creation time: "
        f"{schedule['creation_time'].min()} seconds"
    )
    print(
        f"Last creation time: "
        f"{schedule['creation_time'].max()} seconds"
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
        "\nPriority accuracy in the selected simulation messages: "
        f"{priority_accuracy:.4f}"
    )


def main():
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
    print(f"Saved output to:\n{OUTPUT_FILE}")
    print("Simulation CSV validation passed.")


if __name__ == "__main__":
    main()
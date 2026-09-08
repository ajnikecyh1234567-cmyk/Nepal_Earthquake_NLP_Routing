import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

import joblib




df = pd.read_csv("messages_with_priority.csv")

print("Data loaded successfully.")
print("Dataset shape:", df.shape)
print(df.head())




X = df["clean_text"].fillna("")
y = df["label"]








X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X,
    y,
    df.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", len(X_train))
print("Testing data size:", len(X_test))








model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])





model.fit(X_train, y_train)

print("\nModel training completed.")








pred_label = model.predict(X_test)








label_accuracy = accuracy_score(y_test, pred_label)
label_macro_f1 = f1_score(y_test, pred_label, average="macro")
label_weighted_f1 = f1_score(y_test, pred_label, average="weighted")

print("\n==============================")
print("Label Classification Results")
print("==============================")
print("Accuracy:", label_accuracy)
print("Macro-F1:", label_macro_f1)
print("Weighted-F1:", label_weighted_f1)

print("\nClassification Report:")
print(classification_report(y_test, pred_label, zero_division=0))






priority_map = {
    "injured_or_dead_people": "high",
    "missing_trapped_or_found_people": "high",

    "displaced_people_and_evacuations": "medium_high",
    "infrastructure_and_utilities_damage": "medium_high",
    "caution_and_advice": "medium_high",

    "donation_needs_or_offers_or_volunteering_services": "medium",
    "other_useful_information": "medium",

    "sympathy_and_emotional_support": "low",
    "not_related_or_irrelevant": "low"
}

priority_score_map = {
    "high": 1,
    "medium_high": 2,
    "medium": 3,
    "low": 4
}

test_df = df.loc[test_idx].copy()

test_df["true_label"] = y_test.values
test_df["predicted_label"] = pred_label

test_df["predicted_priority_level"] = test_df["predicted_label"].map(priority_map)
test_df["predicted_priority_score"] = test_df["predicted_priority_level"].map(priority_score_map)

test_df["label_correct"] = test_df["true_label"] == test_df["predicted_label"]
test_df["priority_correct"] = test_df["priority_score"] == test_df["predicted_priority_score"]






priority_accuracy = accuracy_score(
    test_df["priority_score"],
    test_df["predicted_priority_score"]
)

priority_macro_f1 = f1_score(
    test_df["priority_score"],
    test_df["predicted_priority_score"],
    average="macro"
)

priority_weighted_f1 = f1_score(
    test_df["priority_score"],
    test_df["predicted_priority_score"],
    average="weighted"
)

print("\n==============================")
print("Priority Classification Results")
print("==============================")
print("Accuracy:", priority_accuracy)
print("Macro-F1:", priority_macro_f1)
print("Weighted-F1:", priority_weighted_f1)

print("\nPriority Classification Report:")
print(classification_report(
    test_df["priority_score"],
    test_df["predicted_priority_score"],
    zero_division=0
))



# 10. Save output files


test_df[[
    "tweet_id",
    "tweet_text",
    "clean_text",
    "true_label",
    "predicted_label",
    "label_correct",
    "priority_level",
    "priority_score",
    "predicted_priority_level",
    "predicted_priority_score",
    "priority_correct"
]].to_csv("step2_classification_results.csv", index=False)

label_report = classification_report(
    y_test,
    pred_label,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(label_report).T.to_csv(
    "step2_label_classification_report.csv"
)

priority_report = classification_report(
    test_df["priority_score"],
    test_df["predicted_priority_score"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(priority_report).T.to_csv(
    "step2_priority_classification_report.csv"
)

joblib.dump(model, "tfidf_logistic_regression_model.joblib")

print("\nFiles saved successfully:")
print("step2_classification_results.csv")
print("step2_label_classification_report.csv")
print("step2_priority_classification_report.csv")
print("tfidf_logistic_regression_model.joblib")
import pandas as pd
import re

df = pd.read_csv("2015_Nepal_Earthquake_en_CF_labeled_data.tsv", sep="\t")

df.columns = [c.strip() for c in df.columns]


df["tweet_id"] = df["tweet_id"].astype(str).str.strip("'\"")


def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["tweet_text"].apply(clean_tweet)

df = df.drop_duplicates(subset=["tweet_text"]).reset_index(drop=True)

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

df["priority_level"] = df["label"].map(priority_map)

priority_score_map = {
    "high": 1,
    "medium_high": 2,
    "medium": 3,
    "low": 4
}

df["priority_score"] = df["priority_level"].map(priority_score_map)

unmapped_labels = df[df["priority_level"].isnull()]["label"].unique()

if len(unmapped_labels) > 0:
    print("Warning: Some labels were not mapped:")
    print(unmapped_labels)
else:
    print("All labels were successfully mapped.")

df[["tweet_id", "tweet_text", "clean_text", "label"]].to_csv(
    "cleaned_nepal_earthquake_data.csv",
    index=False
)

df[["tweet_id", "tweet_text", "clean_text", "label", "priority_level", "priority_score"]].to_csv(
    "messages_with_priority.csv",
    index=False
)

summary = (
    df.groupby(["label", "priority_level", "priority_score"])
      .size()
      .reset_index(name="count")
      .sort_values(["priority_score", "count"], ascending=[True, False])
)

summary.to_csv("label_priority_summary.csv", index=False)

print("Number of rows:", len(df))
print("Number of labels:", df["label"].nunique())
print(summary)
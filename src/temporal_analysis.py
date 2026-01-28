import pandas as pd

df = pd.read_csv("data/decision_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(["user_id", "timestamp"])

df["rolling_accuracy"] = (
    df.groupby("user_id")["is_correct"]
    .rolling(window=5)
    .mean()
    .reset_index(level=0, drop=True)
)

df.to_csv("data/decision_data_temporal.csv", index=False)

print("Temporal features added.")

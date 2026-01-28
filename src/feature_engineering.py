import pandas as pd
import numpy as np

df = pd.read_csv("data/decision_data.csv")

features = []

for user_id, user_df in df.groupby("user_id"):
    avg_time = user_df["decision_time"].mean()
    time_var = user_df["decision_time"].var()
    accuracy = user_df["is_correct"].mean()

    fast_wrong = ((user_df["decision_time"] < 1.0) & (user_df["is_correct"] == 0)).mean()
    slow_correct = ((user_df["decision_time"] > 3.0) & (user_df["is_correct"] == 1)).mean()

    speed_consistency = 1 / (1 + np.std(user_df["decision_time"]))

    features.append({
        "user_id": user_id,
        "avg_decision_time": avg_time,
        "decision_time_variance": time_var,
        "accuracy_rate": accuracy,
        "fast_but_wrong_freq": fast_wrong,
        "slow_but_correct_freq": slow_correct,
        "speed_consistency_index": speed_consistency
    })

features_df = pd.DataFrame(features)
features_df.to_csv("data/user_features.csv", index=False)

print("User feature table created:", features_df.shape)

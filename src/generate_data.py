import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -------------------------------
# Configuration
# -------------------------------
NUM_USERS = 50
DECISIONS_PER_USER = 40
START_TIME = datetime.now()

np.random.seed(42)

records = []

# -------------------------------
# Data Simulation
# -------------------------------
for user_id in range(1, NUM_USERS + 1):
    base_speed = np.random.uniform(1.5, 4.5)  # personal thinking speed
    
    for d in range(1, DECISIONS_PER_USER + 1):
        task_difficulty = np.random.uniform(0, 1)
        risk_level = np.random.uniform(0, 1)
        reward_value = np.random.uniform(0, 1)
        time_pressure = np.random.uniform(0, 1)

        # decision time influenced by difficulty + pressure
        decision_time = (
            base_speed
            + task_difficulty * 2
            - time_pressure * 1.5
            + np.random.normal(0, 0.4)
        )
        decision_time = max(0.3, decision_time)

        # correctness probability
        correctness_prob = (
            0.75
            - task_difficulty * 0.4
            - risk_level * 0.2
            - (decision_time < 1.0) * 0.2
        )
        correctness_prob = np.clip(correctness_prob, 0.05, 0.95)

        is_correct = np.random.rand() < correctness_prob

        records.append({
            "user_id": f"U{user_id:02d}",
            "decision_id": d,
            "timestamp": START_TIME + timedelta(minutes=len(records)),
            "task_difficulty": round(task_difficulty, 2),
            "risk_level": round(risk_level, 2),
            "reward_value": round(reward_value, 2),
            "time_pressure": round(time_pressure, 2),
            "decision_time": round(decision_time, 2),
            "chosen_option": np.random.choice(["A", "B", "C"]),
            "is_correct": int(is_correct)
        })

# -------------------------------
# Save Dataset
# -------------------------------
df = pd.DataFrame(records)
df.to_csv("data/decision_data.csv", index=False)

print("Dataset generated:", df.shape)
print(df.head())

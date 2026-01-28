import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/decision_data.csv")

X = df[[
    "decision_time",
    "task_difficulty",
    "risk_level",
    "time_pressure"
]]

y = df["is_correct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

print("Model R²:", model.score(X_test, y_test))

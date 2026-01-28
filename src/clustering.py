import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load user features
df = pd.read_csv("data/user_features.csv")

# Drop non-feature column
X = df.drop(columns=["user_id"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df["decision_style"] = kmeans.fit_predict(X_scaled)

# Add human-readable labels
style_map = {
    0: "Fast & Risky",
    1: "Slow & Accurate",
    2: "Balanced",
    3: "Inconsistent"
}
df["decision_style_label"] = df["decision_style"].map(style_map)

# Save results with labels
df.to_csv("data/user_clusters.csv", index=False)

# Print cluster averages for numeric columns
numeric_cols = df.select_dtypes(include="number").columns
print(df.groupby("decision_style")[numeric_cols].mean())

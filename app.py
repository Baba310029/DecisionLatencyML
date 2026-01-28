import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DecisionLatencyML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("🧠 DecisionLatencyML")
st.markdown(
    """
    ### Thinking Speed vs Decision Quality  
    This dashboard explores how **decision-making speed** influences **decision quality**
    under varying conditions such as risk, difficulty, and time pressure.
    """
)


# Load data
df_users = pd.read_csv("data/user_clusters.csv")
df_decisions = pd.read_csv("data/decision_data.csv")

STYLE_COLORS = {
    "Fast & Risky": "#e74c3c",      # red
    "Slow & Accurate": "#27ae60",   # green
    "Balanced": "#2980b9",          # blue
    "Inconsistent": "#f39c12"       # orange
}

st.markdown("## 📊 Overall Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Users", df_users["user_id"].nunique())
col2.metric("Total Decisions", df_decisions.shape[0])
col3.metric("Avg Decision Time (s)", round(df_decisions["decision_time"].mean(), 2))
col4.metric("Overall Accuracy", round(df_decisions["is_correct"].mean() * 100, 1))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
st.sidebar.caption(
    "Filter users by decision-making style to explore behavioral patterns."
)
selected_style = st.sidebar.multiselect(
    "Select Decision Style",
    df_users["decision_style_label"].unique(),
    default=df_users["decision_style_label"].unique()
)

filtered_users = df_users[df_users["decision_style_label"].isin(selected_style)]

# -----------------------------
# Section 1: Overview
# -----------------------------
st.markdown("---")
st.header("🧠 Decision Style Distribution")
st.caption("How users are grouped based on their decision-making behavior.")

fig1, ax1 = plt.subplots()
sns.countplot(
    data=filtered_users,
    x="decision_style_label",
    palette=STYLE_COLORS,
    ax=ax1
)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20)
st.pyplot(fig1)

# -----------------------------
# Section 2: Speed vs Accuracy
# -----------------------------
st.markdown("---")
st.header("⚡ Speed vs Accuracy Trade-off")
st.caption("Each point represents a user. Colors indicate decision-making style.")

fig2, ax2 = plt.subplots()
sns.scatterplot(
    data=filtered_users,
    x="avg_decision_time",
    y="accuracy_rate",
    hue="decision_style_label",
    palette=STYLE_COLORS,
    s=120,
    alpha=0.85,
    ax=ax2
)
ax2.set_xlabel("Average Decision Time (seconds)")
ax2.set_ylabel("Accuracy Rate")
st.pyplot(fig2)

st.info(
    "🔍 Insight: Users with extremely fast decision times tend to show lower accuracy, "
    "while overly slow decision-makers do not always gain proportional accuracy. "
    "Balanced decision speed often yields optimal performance."
)

st.markdown("---")
st.header("👤 Individual User Analysis")

selected_user = st.selectbox(
    "Select a User",
    df_users["user_id"].unique()
)

user_data = df_decisions[df_decisions["user_id"] == selected_user]
user_profile = df_users[df_users["user_id"] == selected_user].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("Decision Style", user_profile["decision_style_label"])
col2.metric("Avg Decision Time (s)", round(user_profile["avg_decision_time"], 2))
col3.metric("Accuracy Rate", round(user_profile["accuracy_rate"] * 100, 1))

# -----------------------------
# Section 3: Raw Decisions
# -----------------------------
st.header("Raw Decision Records (Sample)")

with st.expander("🔎 View Raw Decision Data (Sample)"):
    st.dataframe(df_decisions.head(50))

st.subheader("⏱ Decision Accuracy Over Time")

fig3, ax3 = plt.subplots()
sns.lineplot(
    data=user_data,
    x="timestamp",
    y="is_correct",
    marker="o",
    ax=ax3
)
ax3.set_ylabel("Correctness (1 = Correct)")
ax3.set_xlabel("Time")
st.pyplot(fig3)

st.caption(
    "This timeline shows how the selected user's decision accuracy evolves over time. "
    "Patterns may indicate learning effects, fatigue, or inconsistent decision behavior."
)

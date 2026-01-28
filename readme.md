# 🧠 DecisionLatencyML  
### Thinking Speed vs Decision Quality

DecisionLatencyML is a behavioral machine learning project that analyzes how **decision-making speed** influences **decision quality** under varying conditions such as task difficulty, risk, reward, and time pressure.

Instead of predicting individual choices, this project focuses on **modeling decision behavior over time** to understand *how people think*, not just *what they choose*.

---

## 🔍 Problem Statement

> **Is faster decision-making always worse, or does optimal decision speed depend on context?**

Human decisions are often judged as either “too fast” or “too slow,” but real performance depends on context.  
This project answers the question using **data-driven behavioral modeling** rather than assumptions or intuition.

---

## 🧪 Dataset Design

Real-world datasets that capture **decision latency + correctness + context** are rare.  
Therefore, this project uses a **carefully designed synthetic dataset** that simulates realistic human decision behavior.

Each decision record includes:
- Decision time (latency)
- Correctness (right / wrong)
- Task difficulty
- Risk level
- Reward value
- Time pressure
- Timestamped decision sequences per user

> ⚠️ Note:  
> The entire ML pipeline is designed so that **real experimental or user data can be plugged in later** without changing the modeling logic.

---

## 🧠 Methodology

### 1️⃣ Data Simulation
- Simulated multi-user decision behavior using controlled probabilistic rules
- Generated sequential, time-stamped decision records per user

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyzed speed vs accuracy relationships
- Studied distributions and correlations between context variables
- Identified early behavioral patterns

### 3️⃣ Behavioral Feature Engineering
Converted raw decisions into **user-level behavioral traits**, including:
- Average decision time
- Decision time variance
- Accuracy rate
- Fast-but-wrong frequency
- Slow-but-correct frequency
- Speed consistency index

These features capture **decision style**, not just outcomes.

### 4️⃣ Unsupervised Learning (Clustering)
- Applied KMeans clustering on behavioral features
- Discovered interpretable decision-making styles:
  - **Fast & Risky**
  - **Slow & Accurate**
  - **Balanced**
  - **Inconsistent**

### 5️⃣ Regression Modeling
- Modeled the relationship between decision speed, difficulty, risk, and quality
- Demonstrated that **optimal decision speed depends on context**, not extremes

### 6️⃣ Temporal Analysis
- Analyzed how decision accuracy changes over time
- Captured learning and fatigue effects using rolling accuracy

### 7️⃣ Interactive Dashboard (Streamlit)
Built a polished Streamlit application featuring:
- High-level summary metrics
- Decision style distribution
- Speed vs accuracy trade-off visualization
- Individual user drill-down
- Temporal accuracy timeline
- Human-readable insights and explanations

---

## 📊 Decision Styles Identified

| Style | Description |
|------|-------------|
| Fast & Risky | Quick decisions with higher error rates |
| Slow & Accurate | Deliberate decisions with high accuracy |
| Balanced | Optimal speed–accuracy trade-off |
| Inconsistent | High variability in speed and outcomes |

---

## 🧠 Key Insight

> Decision quality is **not maximized by extreme speed or extreme slowness**.  
> Balanced decision-making often yields the best performance, and optimal speed varies across individuals and contexts.

---

## 🛠 Tech Stack

- **Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn  
- **Visualization**: Matplotlib, Seaborn  
- **App Framework**: Streamlit  

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/feature_engineering.py
python src/clustering.py
streamlit run app.py

# Behavioural Portfolio Optimizer

**Project Type:** FinTech / Behavioral Finance  
**Timeline:** 15 Days  
**Client:** Zetheta

## 📌 Project Overview
This project implements a **Behavioural Portfolio Optimizer** that merges Modern Portfolio Theory (MPT) with Behavioral Finance. Unlike traditional models that assume investors are rational, this system detects cognitive biases (e.g., Loss Aversion, Overconfidence) and adjusts portfolio allocations to protect the user from irrational decision-making.

## 🚀 Features (Deliverables)

1.  **Bias Detection Engine:** Analyzes user inputs to detect psychological biases.
2.  **Bias-Adjusted MPT:** Modifies mathematical constraints (bounds/weights) in the optimization algorithm based on detected biases.
3.  **Nudge System:** Provides actionable feedback to the user.
4.  **Interactive Dashboard:** A Streamlit-based UI for real-time interaction.

## 🛠️ Technology Stack
* **Python 3.9+**
* **Streamlit:** For Frontend Dashboard.
* **Scikit-Learn:** For predictive modeling logic.
* **SciPy:** For mathematical optimization (SLSQP solver).
* **Yfinance:** For fetching real-time historical stock data.

## ⚙️ How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the application:
    ```bash
    streamlit run app.py
    ```

## 📊 Methodology
The system uses a modified Mean-Variance Optimization approach. If a user is flagged with "Loss Aversion," the optimizer imposes stricter upper bounds on volatile assets, forcing a more diversified safety net, effectively "nudging" the user toward a mathematically safer portfolio.
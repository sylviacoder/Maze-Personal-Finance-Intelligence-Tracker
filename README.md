# Maze-Personal-Finance-Intelligence-Tracker
Maze is a Personal Finance Intelligence Tracker designed to help individuals take control of their financial life. The system tracks income, expenses, and savings, identifies spending patterns, and provides actionable insights to improve financial health. Maze is an end-to-end fintech intelligence suite designed to redefine how financial health is measured. By moving beyond traditional, static credit scores, Maze leverages behavioral patterns such as transaction frequency, investment habits, and cash flow volatility to provide real-time predictions for Credit Risk, Financial Stress, and Savings Goals.

Current Status: 🚧 This project is in active development. The core analytical engine and machine learning models are finalized, while the Streamlit deployment and UI integration are currently underway.

## Project Highlights
Hybrid Intelligence Architecture: A unique blend of Machine Learning (XGBoost) for Credit and Savings, combined with a Rule-Based Logic engine for Financial Stress.

Behavioral Scoring: Uses non-traditional features like transaction_frequency, emergency_fund_ratios, and essential_vs_luxury_spend.

Production-Ready: Fully deployable via Streamlit with a focus on real-time inference and a clean, intuitive UX.

End-to-End Pipeline: Integrated Scikit-Learn ColumnTransformers to handle preprocessing and model execution in a single flow.

## The "Financial Stress" Challenge
During development, the initial dataset for the Financial Stress target was found to be statistically inconsistent and "faulty," failing to capture true liquidity pressures.

To maintain the integrity of the suite, we pivoted from a pure ML approach to a Rule-Based Logic Engine for this specific module. This ensured that stress levels are calculated based on verifiable financial thresholds, such as:

The Income-Expense Gap: Real-time monitoring of monthly burn rates.

Liquidity Ratios: Analyzing the ratio of liquid assets to immediate debt obligations.

Behavioral Red Flags: Identifying high-frequency, small-value transactions that often signal cash flow volatility.

## Tech Stack
Language: Python 3.12

Environment: Windows/Anaconda

Machine Learning: XGBoost, Scikit-Learn

Logic Engine: Custom Python-based Rule Logic

Deployment: Streamlit (In Progress)

Serialization: Joblib

## Roadmap
[x] Data Processing: Robust pipeline development using ColumnTransformer.

[x] Model Training: Optimized XGBoost models for Credit Risk and Savings.

[x] Logic Implementation: Development of the rule-based Financial Stress engine.

[ ] Streamlit UI: Designing the user interface for interactive financial tracking.

[ ] Final Deployment: Launching the suite on Streamlit Community Cloud.

## Getting Started
### Environment Setup
It is recommended to use the Anaconda distribution:   

conda create -n maze_env python=3.12
conda activate maze_env

### Installation
Clone the repository:
git clone https://github.com/sylviacoder/Maze-Personal-Finance-Intelligence-Tracker.git
cd Maze-Personal-Finance-Intelligence-Tracker

Install dependencies:
pip install -r requirements.txt

## Documentation & Insights
Detailed findings on behavioral financial patterns and the rationale behind the hybrid logic approach are documented in the PowerPoint presentation located in the docs/ directory.

## License
This project is licensed under the MIT License.

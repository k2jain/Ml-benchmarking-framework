# ML Model Benchmarking & Optimization Framework

Built an automated benchmarking pipeline comparing Logistic Regression, Random Forest, XGBoost, and Neural Networks using Optuna hyperparameter tuning.

## Results
- F1-score: 0.986
- PR-AUC: 0.997
- Latency: ~2.7 ms

## Key Insight
A key finding was that simpler models (Logistic Regression) outperformed more complex models like XGBoost and Neural Networks while achieving 2–4× lower inference latency.

This highlights the importance of balancing performance with efficiency in production ML systems.

## Run
pip install -r requirements.txt
python benchmark.py

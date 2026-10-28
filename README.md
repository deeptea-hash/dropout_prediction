# Predictive Analytics for E-Learning Dropout Rates

**BCA Final Year Project** · School of Computer Applications and Technology, Galgotias University, Greater Noida · Session 2026–2027

| Team member | Enrolment no. |
| --- | --- |
| Yatharth Vardhan Gupta | 24SCSE1500003 |
| Dipti Rani | 24SCSE1480045 |
| Annu Chaudhary | 24SCSE1480009 |

**Guide:** Mr. Kuldeep Kumar

## Overview
A machine-learning workflow that estimates whether an e-learner is likely to **complete or discontinue** a course, using
behavioural and academic indicators (login frequency, recency of activity, assessment average, assignment submission rate,
resource interactions, discussion participation). It supports early, *supportive* intervention and aligns with
**UN Sustainable Development Goal 4 (Quality Education)**.

Pipeline: validate data -> data-quality summary -> EDA charts -> feature engineering -> stratified train/test split ->
train Logistic Regression, Decision Tree, Random Forest, SVM (+ majority-class baseline) -> evaluate on held-out data
(accuracy, precision, recall, F1, confusion matrix) -> risk bands (Low / Medium / High) -> exported report.

Key design rules: preprocessing is fitted on **training data only** (no leakage); dropout is the **positive class**;
bad input (missing columns, invalid labels, single class) stops with a clear error.

## Project structure
```
dropout-prediction/
├── dropout_prediction/      # source package
│   ├── pipeline.py          #   FR1-FR9: load, validate, EDA, preprocess, train, evaluate, predict, report
│   └── demo_data.py         #   SYNTHETIC data generator (demo only)
├── notebooks/
│   └── dropout_prediction.ipynb   # same pipeline, step by step (for screenshots)
├── tests/test_pipeline.py   # automated test cases TC01-TC06 (+ leakage and prediction checks)
├── data/                    # put your real dataset here (not committed)
├── outputs/                 # generated charts, metrics, report (not committed)
├── docs/                    # GitHub + submission guides
├── run_analysis.py          # command-line entry point
├── requirements.txt
└── LICENSE
```

## Setup
Requires Python 3.10+.
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# 1. Demo on SYNTHETIC data (verifies the code runs; results are NOT real findings)
python run_analysis.py --demo

# 2. Real dataset
python run_analysis.py --data data/your_dataset.csv --target dropout_status --name "Dataset name + citation"

# 3. Tests
python -m unittest discover -s tests -t . -v
```
Open `notebooks/dropout_prediction.ipynb` in VS Code, choose the `.venv` kernel and click **Run All**.

Outputs are written to `outputs/`: `report.md`, `model_metrics.csv`, `cross_validation.csv`, `run_summary.json`,
`predictions_test_set.csv` and `figures/`.

## Results
> **To be completed after running on the real dataset.** Dataset: _name, source, size_. Best model: _name_.
> Insert the measured table from `outputs/model_metrics.csv` here. Do not copy numbers from `--demo`: they come from simulated data.

## Ethical use
Predictions are estimates, not certainties. They should trigger supportive review, never penalties. Use learner data only
with permission, minimise identifiers, and check performance across groups where lawful and appropriate.

## Limitations
Depends on dataset quality and label definition; class imbalance; results may not transfer across courses or platforms;
activity logs cannot capture personal circumstances. Not validated for high-stakes institutional decisions.

## License
MIT - see [LICENSE](LICENSE).

# Data folder
Datasets are **not committed to GitHub** (size/licensing). Put your real dataset here, e.g. `data/your_dataset.csv`.

Required columns (see `DEFAULT_FEATURES` in `dropout_prediction/pipeline.py`):
`login_frequency, days_since_last_activity, assessment_average, assignment_submission_rate,
resource_interactions, discussion_participation` plus the target `dropout_status`
(0/1, yes/no, dropout/completed, withdrawn/pass are accepted). One row per learner.
Different column names? Use `--features` / `--target`, or rename the columns.
Remove any column recorded after the prediction time (data leakage).

Record the dataset **name, source/URL, licence, number of records, time period and target definition**
in the project report (Section 3.2).

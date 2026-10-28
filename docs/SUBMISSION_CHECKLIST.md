# Final submission checklist

## Report (Word/PDF)
- [ ] Replace `[STUDENT NAME]` / `[ROLL NUMBER]` in the certificate; fix the stray "ss" after "Student Name:" in the declaration.
- [ ] Declaration/acknowledgement wording matches the team (three authors, not "I").
- [ ] Section 3.2: real dataset name, source/URL, licence, size, period, target definition.
- [ ] Table 4.3 filled with measured values from `outputs/model_metrics.csv`.
- [ ] Section 4.4.1: six real screenshots (dataset preview, missing values, EDA, training output, confusion matrix, metric comparison).
- [ ] Sections 4.5, 5.2, 5.3: write real findings (best model, precision/recall trade-off, comparison with baseline).
- [ ] References: add the dataset citation + any papers actually read, in the university citation style.
- [ ] Confirm formatting, plagiarism and AI-use rules with your guide.

## Code / repository
- [ ] `pip install -r requirements.txt` works in a fresh virtual environment.
- [ ] `python -m unittest discover -s tests -t . -v` -> all tests pass.
- [ ] `python run_analysis.py --data data/<your file> ...` runs on the REAL dataset.
- [ ] Notebook runs top to bottom with `USE_DEMO = False`.
- [ ] README "Results" section filled in; team/guide details correct.
- [ ] Pushed to GitHub; repository link added to the report.

## Be ready to explain (viva)
- Why preprocessing is fitted on training data only (leakage).
- Why accuracy alone misleads with imbalance; what recall means for the dropout class.
- How each model works and why you chose the final one.
- What the demo (synthetic) data is and why its results are not reported as findings.

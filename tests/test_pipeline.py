"""Automated versions of test cases TC01-TC06 from Table 4.2 of the report."""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dropout_prediction import pipeline as dp
from dropout_prediction.demo_data import generate


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate(600, seed=1)

    def test_tc01_valid_csv_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            self.df.to_csv(p, index=False)
            loaded = dp.load_dataset(p)
        out = dp.validate_dataset(loaded)
        self.assertEqual(len(out), len(self.df))
        self.assertTrue(set(out["dropout_status"].unique()) <= {0, 1})

    def test_tc02_missing_target_reports_clear_error(self):
        with self.assertRaisesRegex(dp.DatasetError, "target column"):
            dp.validate_dataset(self.df.drop(columns=["dropout_status"]))
        with self.assertRaisesRegex(dp.DatasetError, "Missing required feature"):
            dp.validate_dataset(self.df.drop(columns=["login_frequency"]))
        bad = self.df.copy()
        bad["dropout_status"] = bad["dropout_status"].astype(object)
        bad.loc[0, "dropout_status"] = "maybe"
        with self.assertRaisesRegex(dp.DatasetError, "Invalid target labels"):
            dp.validate_dataset(bad)

    def test_tc03_missing_feature_values_are_handled(self):
        self.assertGreater(self.df.isna().sum().sum(), 0)
        with tempfile.TemporaryDirectory() as d:
            res = dp.run_pipeline(self.df, d, make_plots=False)
        self.assertFalse(res.metrics.isna().any().any())

    def test_tc04_duplicates_reported_and_removed(self):
        q = dp.data_quality_summary(self.df)
        self.assertGreater(q["duplicate_rows"], 0)
        clean, n = dp.remove_duplicates(self.df)
        self.assertEqual(n, q["duplicate_rows"])
        self.assertEqual(clean.duplicated().sum(), 0)

    def test_tc05_single_class_stops_with_explanation(self):
        one = self.df.copy()
        one["dropout_status"] = 0
        with self.assertRaisesRegex(dp.DatasetError, "single class"):
            dp.run_pipeline(one, tempfile.mkdtemp(), make_plots=False)

    def test_tc06_metrics_and_confusion_matrix_from_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            res = dp.run_pipeline(self.df, d, make_plots=True)
            for col in ("Accuracy", "Precision", "Recall", "F1-score"):
                self.assertIn(col, res.metrics.columns)
            self.assertTrue(any("confusion_matrix" in f for f in res.figures))
            self.assertTrue(os.path.exists(res.report_path))
            _, cms = dp.evaluate_models(res.fitted, res.split)
            for cm in cms.values():
                self.assertEqual(cm.sum(), len(res.split.y_test))  # held-out only

    def test_no_leakage_preprocessor_fit_on_train_only(self):
        split = dp.split_data(dp.validate_dataset(self.df), dp.DEFAULT_FEATURES)
        pipe = dp.Pipeline([("prep", dp.build_preprocessor(split.X_train)),
                            ("m", dp.LogisticRegression(max_iter=500))])
        pipe.fit(split.X_train, split.y_train)
        scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
        self.assertTrue(np.allclose(
            scaler.mean_[0],
            split.X_train["login_frequency"].fillna(split.X_train["login_frequency"].median()).mean()))

    def test_predict_risk_output(self):
        with tempfile.TemporaryDirectory() as d:
            res = dp.run_pipeline(self.df, d, make_plots=False)
        out = dp.predict_risk(res.fitted[res.best_model], res.split.X_test)
        self.assertTrue(set(out["risk_category"]) <= {"Low", "Medium", "High"})
        self.assertTrue(out["dropout_probability"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

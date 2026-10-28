"""
Command-line entry point.

  python run_analysis.py --demo                       # synthetic demo data
  python run_analysis.py --data data/my_dataset.csv \
        --target dropout_status --name "OULAD (Kuzilek et al., 2017)"
"""
import argparse
import sys
import warnings

from dropout_prediction import pipeline as dp
from dropout_prediction.demo_data import generate


warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description="E-learning dropout prediction")
    ap.add_argument("--data", help="path to labelled CSV/XLSX dataset")
    ap.add_argument("--demo", action="store_true", help="use SYNTHETIC demo data")
    ap.add_argument("--target", default=dp.TARGET_DEFAULT)
    ap.add_argument("--features", nargs="*", default=dp.DEFAULT_FEATURES,
                    help="feature columns (default: report Table 3.1)")
    ap.add_argument("--name", default=None, help="dataset name/citation for the report")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    if a.demo:
        df, name, synth = generate(), "SYNTHETIC demo data (not real learners)", True
    elif a.data:
        df, name, synth = dp.load_dataset(a.data), a.name or a.data, False
    else:
        ap.error("provide --data FILE or --demo")

    try:
        res = dp.run_pipeline(df, a.out, a.features, a.target, name, synth)
    except dp.DatasetError as e:
        print(f"DATASET ERROR: {e}", file=sys.stderr)
        return 2

    print(f"\nDataset: {name}")
    print(f"Quality: {res.quality}\n")
    print("Held-out test metrics (positive class = dropout):")
    print(res.metrics.to_string())
    print(f"\nBest model by F1: {res.best_model}")
    print(f"Report:  {res.report_path}")
    print(f"Figures: {len(res.figures)} saved in {a.out}/figures")
    if synth:
        print("\nNOTE: results are from SYNTHETIC data - do not report them as real findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

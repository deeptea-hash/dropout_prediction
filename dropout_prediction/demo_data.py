"""
Generate a SYNTHETIC e-learning dataset so the pipeline can be demonstrated
when no real dataset is available.

IMPORTANT: this data is simulated. Results obtained from it only prove the code
runs; they are NOT findings about real learners and must not be presented as
experimental results in the report. Replace it with a real, cited dataset
(e.g. OULAD, a Kaggle MOOC dropout dataset, or institutional data).
"""
import argparse
import numpy as np
import pandas as pd


def generate(n: int = 2000, seed: int = 42, dropout_rate: float = 0.30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # latent "engagement" drives both behaviour and outcome (with noise)
    engagement = rng.normal(0, 1, n)
    login = np.clip(rng.poisson(np.exp(2.4 + 0.45 * engagement)), 0, None)
    days_since = np.clip(rng.exponential(np.exp(1.8 - 0.55 * engagement)), 0, 90).round()
    assess = np.clip(62 + 14 * engagement + rng.normal(0, 10, n), 0, 100).round(1)
    submit = np.clip(0.72 + 0.17 * engagement + rng.normal(0, 0.12, n), 0, 1).round(3)
    resources = np.clip(rng.poisson(np.exp(3.0 + 0.5 * engagement)), 0, None)
    discuss = np.clip(rng.poisson(np.exp(0.9 + 0.5 * engagement)), 0, None)

    # dropout probability: logistic in engagement + noise, calibrated to target rate
    logit = -1.6 * engagement + rng.normal(0, 0.9, n)
    thresh = np.quantile(logit, 1 - dropout_rate)
    p = 1 / (1 + np.exp(-(logit - thresh) * 2.2))
    drop = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "learner_id": [f"L{i:05d}" for i in range(n)],
            "login_frequency": login,
            "days_since_last_activity": days_since,
            "assessment_average": assess,
            "assignment_submission_rate": submit,
            "resource_interactions": resources,
            "discussion_participation": discuss,
            "dropout_status": drop,
        }
    )
    # realistic data-quality problems (handled by the pipeline)
    for col, frac in [("assessment_average", 0.04), ("discussion_participation", 0.06)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)  # duplicates
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--out", default="data/synthetic_elearning.csv")
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    generate(a.rows).to_csv(a.out, index=False)
    print(f"Wrote {a.out} (SYNTHETIC demo data)")

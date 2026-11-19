
from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible synthetic research dataset
rng = np.random.default_rng(42)
N = 5000

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "synthetic_pregnancy_data.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

regions = [
    "Dar es Salaam", "Dodoma", "Morogoro",
    "Mwanza", "Arusha"
]

facility_ids = [f"HF{i:03d}" for i in range(1, 31)]

df = pd.DataFrame({
    "pregnancy_id": [f"PRG{i:05d}" for i in range(1, N + 1)],
    "facility_id": rng.choice(facility_ids, N),
    "region": rng.choice(regions, N),
    "maternal_age": np.clip(
        rng.normal(28, 6, N).round(), 15, 45
    ).astype(int),
    "parity": rng.poisson(1.7, N).clip(0, 8),
    "gestational_age_weeks": rng.integers(20, 33, N),
    "anc_visits": rng.poisson(3, N).clip(0, 10),
    "bmi": np.clip(rng.normal(25.5, 4.8, N), 16, 45),
    "previous_hypertension": rng.binomial(1, 0.11, N),
    "previous_adverse_outcome": rng.binomial(1, 0.13, N),
})

# Generate correlated clinical measurements
df["systolic_bp"] = (
    112
    + 0.35 * (df["maternal_age"] - 28)
    + 9 * df["previous_hypertension"]
    + 0.55 * (df["bmi"] - 25)
    + rng.normal(0, 12, N)
).clip(85, 190).round()

df["diastolic_bp"] = (
    73
    + 0.25 * (df["maternal_age"] - 28)
    + 6 * df["previous_hypertension"]
    + 0.35 * (df["bmi"] - 25)
    + rng.normal(0, 8, N)
).clip(50, 125).round()

df["haemoglobin_g_dl"] = np.clip(
    rng.normal(11.2, 1.5, N), 6.5, 16
).round(1)

# Illustrative, invented data-generating mechanism.
# These coefficients are NOT clinical risk estimates.
linear_score = (
    -3.5
    + 0.045 * (df["maternal_age"] - 28)
    + 0.035 * (df["bmi"] - 25)
    + 0.045 * (df["systolic_bp"] - 115)
    + 0.025 * (df["diastolic_bp"] - 75)
    + 0.9 * df["previous_hypertension"]
    + 0.4 * df["previous_adverse_outcome"]
)

simulated_probability = 1 / (1 + np.exp(-linear_score))

df["hypertensive_complication"] = rng.binomial(
    1, simulated_probability.to_numpy()
)

# Add some missing values for data-quality exercises
for column, missing_rate in {
    "bmi": 0.05,
    "haemoglobin_g_dl": 0.07,
    "anc_visits": 0.03,
}.items():
    missing_rows = rng.choice(
        N, size=int(N * missing_rate), replace=False
    )
    df.loc[missing_rows, column] = np.nan

df.to_csv(OUTPUT, index=False)

print("Dataset generated successfully")
print(f"Saved to: {OUTPUT}")
print(f"Records: {len(df):,}")
print(f"Facilities: {df['facility_id'].nunique()}")
print(
    "Simulated complication prevalence:",
    f"{df['hypertensive_complication'].mean():.2%}"
)
print("\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0])
print("\nFirst five records:")
print(df.head().to_string(index=False))

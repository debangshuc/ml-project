"""
K-Means Customer Segmentation - Data Preparation
=================================================
Cleans the Mall Customers dataset and prepares model-ready inputs.
NO MODEL TRAINING HAPPENS HERE - this only produces:
  1. data/Mall_Customers_clean.csv   - cleaned dataset
  2. data/features_scaled.csv        - scaled feature matrix ready for KMeans
  3. models/scaler.pkl               - fitted StandardScaler (reusable at inference)
  4. output/preparation_report.txt   - full audit log of what was done
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent
RAW_PATH = PROJECT_ROOT / "Mall_Customers.csv"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Columns used for the classic customer-segmentation demo
CLUSTER_FEATURES = ["annual_income", "spending_score"]

report_lines = []


def log(msg: str = "") -> None:
    print(msg)
    report_lines.append(msg)


# ---------------------------------------------------------------- 1. Load ---
log("=" * 62)
log("STEP 1: LOAD RAW DATA")
log("=" * 62)
df = pd.read_csv(RAW_PATH)
log(f"Loaded: {RAW_PATH.name}  |  shape = {df.shape}")
log(f"Columns: {list(df.columns)}")

# ------------------------------------------------------------- 2. Validate ---
log()
log("=" * 62)
log("STEP 2: DATA QUALITY AUDIT")
log("=" * 62)

issues = []

# 2a. Schema check - expected raw columns
expected_cols = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]
if list(df.columns) != expected_cols:
    issues.append(f"Unexpected schema: {list(df.columns)}")
log(f"Schema matches expected Kaggle layout: {list(df.columns) == expected_cols}")

# 2b. Missing values
n_missing = int(df.isna().sum().sum())
if n_missing:
    issues.append(f"{n_missing} missing values need handling")
log(f"Missing values: {n_missing} (per column: {df.isna().sum().to_dict()})")

# 2c. Duplicates
n_dupes = int(df.duplicated().sum())
if n_dupes:
    df = df.drop_duplicates().reset_index(drop=True)
    issues.append(f"{n_dupes} duplicate rows dropped")
log(f"Duplicate rows: {n_dupes} -> dropped and reindexed" if n_dupes else "Duplicate rows: 0")

# 2d. CustomerID integrity
if not df["CustomerID"].is_unique:
    issues.append("CustomerID is not unique")
log(f"CustomerID unique: {df['CustomerID'].is_unique}")

# 2e. Value-range sanity checks (catch corrupt/ impossible readings)
ranges = {"Age": (0, 120), "Annual Income (k$)": (0, 10000), "Spending Score (1-100)": (0, 100)}
for col, (lo, hi) in ranges.items():
    out_of_range = int(((df[col] < lo) | (df[col] > hi)).sum())
    if out_of_range:
        issues.append(f"{col}: {out_of_range} out-of-range values")
    log(f"Range check {col}: {out_of_range} violations "
        f"(observed {df[col].min()}-{df[col].max()}, allowed {lo}-{hi})")

# 2f. Categorical domain check
bad_gender = ~df["Gender"].isin(["Male", "Female"])
if bad_gender.any():
    issues.append(f"{int(bad_gender.sum())} invalid Gender labels")
log(f"Gender domain: {df['Gender'].value_counts().to_dict()} (invalid: {int(bad_gender.sum())})")

# 2g. Quick outlier scan (IQR rule, info only - no removal)
log()
log("Outlier scan (IQR fence 1.5x) - informational only:")
for col in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    vals = df.loc[mask, col].tolist()
    note = f" -> {vals}" if vals else " (none)"
    log(f"  {col}: {int(mask.sum())} potential outlier(s){note}")
log("Decision: keep all values - they are plausible real customers, "
    "and K-Means on income/spend should see genuine high spenders.")

# 2h. Final audit verdict
log()
if issues:
    log(f"AUDIT ISSUES FOUND: {issues}")
else:
    log("AUDIT RESULT: PASS - dataset is complete, unique, and within valid ranges.")

# ---------------------------------------------------------------- 3. Clean ---
log()
log("=" * 62)
log("STEP 3: CLEANING / TRANSFORMATIONS")
log("=" * 62)

# 3a. Drop CustomerID - pure row identifier, carries no behavioural signal
df = df.drop(columns=["CustomerID"])
log("Dropped 'CustomerID' (identifier, no clustering signal)")

# 3b. Standardise column names to snake_case for clean downstream code
df = df.rename(columns={
    "Gender": "gender",
    "Age": "age",
    "Annual Income (k$)": "annual_income",
    "Spending Score (1-100)": "spending_score",
})
log(f"Standardised column names -> {list(df.columns)}")

# 3c. Encode gender (kept alongside features for optional extended experiments;
#     the main clustering set is income + spending score, which are numeric)
df["gender_encoded"] = df["gender"].map({"Male": 0, "Female": 1})
log("Encoded 'gender' -> 'gender_encoded' (Male=0, Female=1)")
log(f"Encoding check (no NaNs introduced): {df['gender_encoded'].isna().sum() == 0}")

# 3d. Enforce dtypes
df["gender_encoded"] = df["gender_encoded"].astype("int64")
log(f"Final dtypes: {df.dtypes.to_dict()}")

# --------------------------------------------------------- 4. Feature prep ---
log()
log("=" * 62)
log("STEP 4: FEATURE MATRIX FOR K-MEANS (scaled, NOT trained)")
log("=" * 62)

X = df[CLUSTER_FEATURES].to_numpy(dtype=float)
log(f"Feature matrix shape: {X.shape} | features = {CLUSTER_FEATURES}")
log(f"Raw feature stats -> income: mean={X[:,0].mean():.2f}, sd={X[:,0].std():.2f}; "
    f"score: mean={X[:,1].mean():.2f}, sd={X[:,1].std():.2f}")

# WHY scale: income (15-137) and score (1-99) live on different ranges;
# unscaled income would dominate euclidean distance in K-Means.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
log("Applied StandardScaler (fit on full dataset - standard for an "
    "unsupervised demo; scaler saved for reuse)")
log(f"Scaled stats  -> income: mean={X_scaled[:,0].mean():.4f}, sd={X_scaled[:,0].std():.4f}; "
    f"score: mean={X_scaled[:,1].mean():.4f}, sd={X_scaled[:,1].std():.4f}")
log(f"Scaler parameters: mean_={scaler.mean_.tolist()}, scale_={scaler.scale_.tolist()}")

# ------------------------------------------------------------------ 5. Save ---
log()
log("=" * 62)
log("STEP 5: SAVE OUTPUTS")
log("=" * 62)

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

clean_path = DATA_DIR / "Mall_Customers_clean.csv"
scaled_path = DATA_DIR / "features_scaled.csv"
scaler_path = MODEL_DIR / "scaler.pkl"
report_path = OUTPUT_DIR / "preparation_report.txt"

df.to_csv(clean_path, index=False)
log(f"Saved cleaned dataset    -> {clean_path.relative_to(PROJECT_ROOT)} ({df.shape[0]} rows)")

pd.DataFrame(
    X_scaled,
    columns=[f"{c}_z" for c in CLUSTER_FEATURES],
).to_csv(scaled_path, index=False)
log(f"Saved scaled features    -> {scaled_path.relative_to(PROJECT_ROOT)} "
    f"({X_scaled.shape[0]} rows x {X_scaled.shape[1]} cols)")

joblib.dump(scaler, scaler_path)
log(f"Saved fitted scaler      -> {scaler_path.relative_to(PROJECT_ROOT)}")

report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
log(f"Saved preparation report -> {report_path.relative_to(PROJECT_ROOT)}")

log()
log("PREPARATION COMPLETE - ready for training (no model was fitted).")
log("Next steps: choose k (elbow / silhouette on features_scaled.csv), then fit KMeans.")

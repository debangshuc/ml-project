"""
K-Means Customer Segmentation - One-File Training + Visualization
=================================================================
Works locally AND in Google Colab unchanged.
If 'Mall_Customers.csv' is missing, it downloads it automatically.

Pipeline:
  1. Load data           (auto-download fallback for Colab)
  2. Prepare features    (select income + spending score, standardize)
  3. Choose k            (elbow method + silhouette scores)
  4. Train KMeans        (k=5, classic customer segments)
  5. Visualize           (elbow, silhouette, cluster scatter, cluster sizes)
"""

from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("Mall_Customers.csv")
OUTPUT_DIR = Path("output")
RANDOM_STATE = 42
CHOSEN_K = 5  # classic segmentation for this dataset - adjust after reading the elbow plot


# ------------------------------------------------------------------ 1. Load ---
def load_data() -> pd.DataFrame:
    """Load raw CSV; in Colab it auto-downloads from the public Kaggle URL."""
    if not DATA_PATH.exists():
        print(f"'{DATA_PATH}' not found - downloading dataset...")
        url = ("https://www.kaggle.com/api/v1/datasets/download/"
               "vjchoudhary7/customer-segmentation-tutorial-in-python")
        zip_path = "_dataset.zip"
        urlretrieve(url, zip_path)  # URL actually returns a zip containing the CSV
        pd.read_csv(zip_path, compression="zip").to_csv(DATA_PATH, index=False)
        Path(zip_path).unlink()
    print(f"Loaded raw data from {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


df_raw = load_data()
print(df_raw.head(), "\n")


# ------------------------------------------------------------- 2. Prepare ---
# Clean: drop the ID, select the two behavioural features, standardize.
# (Income 15-137 vs Score 1-99 - without scaling, income would dominate
#  the Euclidean distance that K-Means relies on.)
X_raw = df_raw[["Annual Income (k$)", "Spending Score (1-100)"]].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"Feature matrix: {X.shape} | features: annual_income, spending_score\n")


# ------------------------------------------------------- 3. Choose k plots ---
k_range = range(1, 11)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels) if k > 1 else float("nan"))

for k, inertia, sil in zip(k_range, inertias, silhouettes):
    print(f"k={k}: inertia={inertia:9.2f}  silhouette={'' if k == 1 else f'{sil:.3f}'}")


# --------------------------------------------------------------- 4. Train ---
kmeans = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=RANDOM_STATE)
df = df_raw.rename(columns={
    "Annual Income (k$)": "annual_income",
    "Spending Score (1-100)": "spending_score",
})
df["cluster"] = kmeans.fit_predict(X)
print(f"\nTrained KMeans with k={CHOSEN_K}")
print(f"Inertia: {kmeans.inertia_:.2f} | "
      f"Silhouette: {silhouette_score(X, df['cluster']):.3f}")

# Cluster summary: which segment spends what
print("\nCluster summary:")
print(df.groupby("cluster").agg(
    n=("cluster", "size"),
    income_mean=("annual_income", "mean"),
    score_mean=("spending_score", "mean"),
).round(1))


# --------------------------------------------------------- 5. Visualize ---
OUTPUT_DIR.mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("K-Means Customer Segmentation", fontsize=15, fontweight="bold")

# (a) Elbow plot
ax = axes[0, 0]
ax.plot(k_range, inertias, "o-", color="steelblue")
CHOSEN_INERTIA = inertias[CHOSEN_K - 1]
ax.axvline(CHOSEN_K, color="red", ls="--", label=f"chosen k={CHOSEN_K}")
ax.set(title="Elbow Method", xlabel="k (number of clusters)",
       ylabel="Inertia (within-cluster SS)")
ax.legend(); ax.grid(alpha=0.3)

# (b) Silhouette plot
ax = axes[0, 1]
ax.plot(list(k_range)[1:], silhouettes[1:], "o-", color="seagreen")
ax.axvline(CHOSEN_K, color="red", ls="--", label=f"chosen k={CHOSEN_K}")
ax.set(title="Silhouette Score vs k", xlabel="k", ylabel="Silhouette")
ax.legend(); ax.grid(alpha=0.3)

# (c) Cluster scatter with centroids (back in original units for readability)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
colors = plt.cm.tab10(df["cluster"] / CHOSEN_K)
ax = axes[1, 0]
for c in range(CHOSEN_K):
    part = df[df["cluster"] == c]
    ax.scatter(part["annual_income"], part["spending_score"],
               s=45, alpha=0.75, label=f"Cluster {c}", color=colors[df["cluster"] == c][0])
ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=250,
           c="black", edgecolors="white", linewidths=1.5, label="Centroids")
ax.set(title=f"Customer Segments (k={CHOSEN_K})",
       xlabel="Annual Income (k$)", ylabel="Spending Score (1-100)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (d) Cluster sizes
ax = axes[1, 1]
sizes = df["cluster"].value_counts().sort_index()
ax.bar(sizes.index, sizes.values, color=plt.cm.tab10(sizes.index / CHOSEN_K))
for i, v in sizes.items():
    ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
ax.set(title="Customers per Cluster", xlabel="Cluster", ylabel="Count")
ax.set_xticks(range(CHOSEN_K)); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plot_path = OUTPUT_DIR / "kmeans_clusters.png"
plt.savefig(plot_path, dpi=130, bbox_inches="tight")
print(f"\nSaved plot -> {plot_path}")
plt.show()  # interactive in Colab / local window

# Persist artifacts for reuse (organized in folders)
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")
df.to_csv("data/customers_with_clusters.csv", index=False)
print("Saved: models/scaler.pkl, models/kmeans_model.pkl, data/customers_with_clusters.csv")

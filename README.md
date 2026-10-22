# Customer Segmentation with K-Means

Ever wondered what mall customers really look like on paper? I clustered 200
mall shoppers by their **annual income** and **spending score** and found five
very different people — including one group of high earners who barely spend at
all (a huge missed opportunity for any marketer).

Built with Python, scikit-learn, and matplotlib. The whole thing is
**one script** (`kmeans_train.py`) — it runs locally or in Google Colab with
zero setup, and even downloads the dataset by itself.

▶️ **Run it yourself in the browser:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tMZTpxrk-4NPRtwDuLZKjph-LQYdDfZw?usp=sharing)

![Clusters](output/kmeans_clusters.png)

## What the clusters say

| Segment | Who they are |
|---|---|
| **Premium spenders** (~$86k income, score 82) | The dream customers — rich *and* spending freely |
| **Careful rich** (~$88k income, score 17) | High earners who hold back — the untapped market |
| **Impulsive buyers** (~$26k income, score 79) | Spend beyond their income — easy wins with promos |
| **Standard** (~$55k income, score 49) | The average middle — everything in moderation |
| **Budget shoppers** (~$26k income, score 21) | Low income, low spend — browse but rarely buy |

## The algorithm — what is K-Means and why it fits here

**K-Means is an unsupervised clustering algorithm** — it is given no labels at
all. It starts by dropping *k* random points (centroids) onto the feature space,
then repeats two steps until nothing moves: (1) assign every customer to the
nearest centroid, and (2) move each centroid to the average position of the
customers assigned to it. When finished, the *k* centroids mark the centers of
k natural groups in the data.

**Why it helps in our case:** the mall has no "labels" like *is a luxury
shopper*. What it does have is two numbers per customer. K-Means is exactly the
tool for that situation — find structure in unlabeled data. Instead of guessing
market segments by intuition, we let the data reveal them, and every customer
gets a precise segment membership.

**Why it's the right tool for *this* data:**

- The data is **numeric and low-dimensional** (2 features) — K-Means works in
  Euclidean space, and with two axes we can literally *see* the clusters, which
  makes it perfect for teaching and demonstration.
- The clusters are expected to be **circular/blobby and roughly equal-sized** —
  exactly the shape assumption K-Means makes (each cluster is modeled by one
  centroid and the distance to it). A glance at the scatter plot shows the
  five groups match this.
- It's **fast and simple** — 200 rows in milliseconds, no hyperparameter maze,
  and the result (a centroid = a "typical customer") is directly explainable to
  a marketing team.

## Choosing k — and why our model is neither under- nor over-fitted

Clustering has no accuracy score, but we still have to worry about the two
classic failures:

**Underfitting = k too small.** With k=1 every customer belongs to one blob and
all structure is lost. Inertia (within-cluster sum of squares) stays huge
(400.0 at k=1) and the silhouette is undefined — the model is too simple to
describe the data.

**Overfitting = k too large.** Given enough clusters, K-Means will happily
carve the noise into fragments: at k=10 inertia keeps falling (29.98) but the
silhouette *drops* to 0.443 — clusters overlap, become uninterpretable, and
stop matching real customer types.

Our k=5 sits in the healthy middle: it sits on the visible **elbow** of the
inertia curve, and the **silhouette score peaks exactly there** (0.555 at k=5
vs 0.540 at 6 and 0.455 at 8). Two independent signals agreeing means we
captured the real structure without chasing noise.

## A few optimization decisions worth knowing

- **Feature standardization** — income (15–137k) and score (1–100) have very
  different ranges. Without `StandardScaler`, income would dominate the
  Euclidean distance and spending behavior — the more interesting dimension —
  would barely matter. Scaled to mean 0 / sd 1, both count equally.
- **`n_init=10`, `random_state=42`** — K-Means is sensitive to where the
  centroids start (bad starts can converge to poor local minima). Running the
  fit 10 times and keeping the best makes results stable, and the fixed seed
  makes them exactly reproducible.
- **Keeping the two "outliers"** — the two $137k earners tripped an IQR scan,
  but trimming them would delete exactly the premium segment we're hunting for.
- **Instant reuse** — the fitted scaler and model are saved, so a new customer
  can be assigned to a segment in one line:
  `kmeans.predict(scaler.transform([[income, score]]))`.

## How to run it

```bash
python kmeans_train.py
```

That's it — locally via Python, or open the **Colab notebook above** and run
the cell (dependencies are preinstalled; the dataset auto-downloads). Want a
different number of clusters? Change `CHOSEN_K` at the top and compare the new
elbow/silhouette plots.

## Data

[Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
— 200 customers, no missing values or duplicates. The full pipeline
(prep → train → visualize) is reproducible end to end with `random_state=42`.

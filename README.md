# Customer Segmentation with K-Means

Ever wondered what mall customers really look like on paper? I clustered 200
mall shoppers by their **annual income** and **spending score** and found five
very different people — including one high earners who barely spend at all
(a huge missed opportunity for any marketer).

Built with Python, scikit-learn, and matplotlib. The whole thing is
**one script** (`kmeans_train.py`) — it runs locally or in
[Google Colab](https://colab.research.google.com/) with zero setup, and even
downloads the dataset by itself.

![Clusters](output/kmeans_clusters.png)

## What the clusters say

| Segment | Who they are |
|---|---|
| **Premium spenders** (~$86k income, score 82) | The dream customers — rich *and* spending freely |
| **Careful rich** (~$88k income, score 17) | High earners who hold back — the untapped market |
| **Impulsive buyers** (~$26k income, score 79) | Spend beyond their income — easy wins with promos |
| **Standard** (~$55k income, score 49) | The average middle — everything in moderation |
| **Budget shoppers** (~$26k income, score 21) | Low income, low spend — browse but rarely buy |

The cluster count (k=5) wasn't picked by eye: the elbow method bends at 5 and
the silhouette score peaks at 5 (0.555), so both checks agree.

## How to run it

```bash
python kmeans_train.py
```

That's it. In Colab, upload the script (or paste it into one cell) and run it —
dependencies are preinstalled and the dataset auto-downloads. Want a different
number of clusters? Change `CHOSEN_K` at the top.

## A few decisions worth knowing

- Features are **standardized** before clustering — raw income (up to 137) would
  otherwise drown out the spending score (up to 100) in the distance calculation.
- Two flagged "outliers" (the 137k earners) were **kept on purpose** — outliers
  in spending data are exactly the premium segment you don't want to delete.
- The fitted model and scaler are saved as pickles, so a new customer can be
  assigned to a segment instantly: `kmeans.predict(scaler.transform([[85, 100]]))`.

## Data

[Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
— 200 customers, no missing values or duplicates. The full pipeline (prep →
train → visualize) is reproducible end to end with `random_state=42`.

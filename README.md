# cluster-trainer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![CI](https://github.com/HSJplayz/cluster-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/HSJplayz/cluster-trainer/actions/workflows/ci.yml)
[![Streak Day 18](https://img.shields.io/badge/Streak-Day%2018-green.svg)](https://github.com/HSJplayz/streak)

k-means clustering and PCA **from scratch** — k-means++ seeding, Lloyd's
algorithm, multiple restarts, the elbow method, silhouette scoring, and SVD-based
dimensionality reduction. JSON model persistence, a CLI, and a web UI with
offline SVG charts. Only numpy (plus Flask for the web app).

Built with numpy on **Day 18** of the daily projects streak.

## Quick start

```bash
pip install -r requirements.txt

# Cluster a CSV into k groups (auto-detects numeric columns)
python cli.py cluster customers.csv -k 4

# Let silhouette choose k, and export labels + the trained model
python cli.py cluster customers.csv --auto-k --k 10 --labels labelled.csv --save model.json

# Inertia + silhouette for k = 1..k_max (the elbow curve, as text)
python cli.py elbow customers.csv --k-max 10

# Project onto the top 2 principal components
python cli.py pca customers.csv -o pca2d.csv

# Self-contained demo (no data needed)
python cli.py demo -k 4

# Web UI
python app.py   # then open http://127.0.0.1:5000
```

## What it does

| Piece | Details |
| --- | --- |
| **Clustering** | `KMeans` with k-means++ init, Lloyd iterations, `n_init` restarts, optional feature standardization, seeded/reproducible |
| **Reduction** | `PCA` via economy SVD — components, singular values, explained variance + ratio |
| **Choosing k** | `elbow` sweep and `best_k_by_silhouette` automatic selection |
| **Quality** | `silhouette_score` (mean coefficient) and `cluster_profile` (per-cluster means/std) |
| **Persistence** | `save_model`/`load_model` to JSON (centroids + scaler, or PCA mean + components) |
| **CLI** | `cluster`, `elbow`, `pca`, `demo` with exit codes 0/1 |
| **Web** | paste CSV → PCA scatter, elbow + silhouette chart, cluster profiles, centers (offline SVG) |

### Example

```
$ python cli.py cluster customers.csv --auto-k -k 6
Auto-k: chose k=3 (silhouette=0.8554)
k-means: 120 rows, 2 feature(s), k=3, standardized=True
Inertia: 8.318   Iterations: 2   Silhouette: 0.8554
Cluster sizes: [40, 40, 40]
Cluster centers (original units):
  cluster 0: x=0.123, y=-0.084
  cluster 1: x=7.904, y=0.238
  cluster 2: x=3.905, y=6.966
```

## Library use

```python
import numpy as np
from clustering import KMeans, PCA, best_k_by_silhouette, cluster_profile, silhouette_score

X, _ = ...  # your (n, d) feature matrix

picked = best_k_by_silhouette(X, k_max=8)
model = KMeans(n_clusters=picked["k"], seed=42).fit(X, feature_names=["a", "b"])
print(model.sizes(), round(silhouette_score(X, model.labels_), 3))
print(cluster_profile(X, model.labels_, ["a", "b"]))

z = PCA(n_components=2).fit_transform(X)   # 2-D projection for plotting
```

## Methods, honestly

- **k-means++** seeds centers proportional to squared distance, which avoids most
  bad local minima; `n_init` restarts still run and the lowest-inertia solution
  wins.
- **Standardization is on by default** because k-means uses Euclidean distance —
  an unscaled column with a large range would dominate. Turn it off with
  `--no-standardize` when features are already comparable.
- **Silhouette is O(n²)** (a full distance matrix), so it is intended for small
  to moderate datasets; the browser computes it off the server for pasted data.
- **PCA** centers the data and takes the top right-singular vectors; it does not
  standardize, so standardize beforehand if columns have wildly different units.
- Cluster ids are arbitrary — a run may permute them relative to another run.

## Project layout

```
clustering.py         KMeans, PCA, silhouette, elbow, profiles, JSON persistence
cli.py                cluster / elbow / pca / demo commands
app.py                Flask JSON API (/health, /api/cluster)
templates/index.html  self-contained UI with inline SVG charts
test_clustering.py    38 unittest cases
```

## Future scope

- DBSCAN and hierarchical (agglomerative) clustering
- Gap statistic and Calinski–Harabasz index for choosing k
- Incremental / mini-batch k-means for large data
- PCA whitening and kernel PCA
- CSV/Excel input helpers and a downloadable cluster report

## Tests

```bash
python -m unittest test_clustering -v
```

38 tests: cluster recovery on separated blobs, inertia monotonicity in k,
seeded reproducibility, PCA variance ordering and orthonormality, silhouette on
separated vs overlapping data, elbow/profile structure, persistence
round-trips, CLI exit codes, and Flask-client API tests. No network.

## License

MIT — see [LICENSE](LICENSE).

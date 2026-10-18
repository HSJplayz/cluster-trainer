# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-18

### Added
- `clustering.py`: from-scratch unsupervised learning on numpy only
  - `KMeans` with k-means++ seeding, Lloyd's algorithm, multiple restarts
    (best inertia wins), optional standardization, and reproducible seeds.
  - `PCA` via economy SVD (components, explained variance + ratio).
  - `silhouette_score` (mean coefficient, O(n²)), `elbow` sweep, and
    `best_k_by_silhouette` automatic k selection.
  - `cluster_profile` for per-cluster feature means/standard deviations.
  - JSON `save_model`/`load_model` with a `_model` registry dispatch.
- `cli.py`: `cluster` (with `--auto-k`, `--labels`, `--save`), `elbow`,
  `pca`, and `demo` commands; exit codes 0/1.
- `app.py` + `templates/index.html`: Flask `/health` and `POST /api/cluster`
  returning chart-ready JSON, with a self-contained UI (PCA scatter, elbow +
  silhouette chart, cluster profile and center tables) rendered as inline SVG.
- `test_clustering.py`: 38 unittest cases covering cluster recovery, inertia
  monotonicity, PCA variance/orthonormality, silhouette behaviour, profiles,
  persistence round-trips, CLI exit codes, and the Flask API. No network.

[1.0.0]: https://github.com/HSJplayz/cluster-trainer/releases/tag/v1.0.0

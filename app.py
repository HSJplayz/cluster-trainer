"""Web front-end for cluster-trainer (Flask). Fits k-means (and PCA) on pasted
CSV and returns chart-ready JSON: PCA scatter, elbow/silhouette curve, cluster
centers, and per-cluster profiles. The UI renders everything as inline SVG."""

from __future__ import annotations

import csv
import io

import numpy as np
from flask import Flask, jsonify, render_template, request

from clustering import (
    KMeans,
    PCA,
    best_k_by_silhouette,
    cluster_profile,
    elbow,
    silhouette_score,
)

app = Flask(__name__)


def _parse_csv(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("CSV has no rows (or headers are missing)")
    return rows


def _numeric_cols(rows: list[dict]) -> list[str]:
    cols = []
    for name in rows[0].keys():
        try:
            [float(r[name]) for r in rows]
            cols.append(name)
        except (TypeError, ValueError):
            continue
    return cols


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/cluster")
def api_cluster():
    try:
        body = request.get_json(force=True, silent=True) or {}
        rows = _parse_csv(body.get("csv", ""))
        features = body.get("features") or []
        standardize = bool(body.get("standardize", True))
        auto_k = bool(body.get("auto_k"))
        seed = int(body.get("seed", 42))
        k_max = int(body.get("k_max", 8))
        k = int(body.get("k", 3))

        if not features:
            features = _numeric_cols(rows)
        if not features:
            raise ValueError("no numeric feature columns found")
        missing = [f for f in features if f not in rows[0]]
        if missing:
            raise ValueError(f"feature column(s) missing: {missing}")
        if len(rows) < 6:
            raise ValueError("need at least 6 rows to cluster")

        X = np.asarray([[float(r[f]) for f in features] for r in rows])
        if auto_k:
            picked = best_k_by_silhouette(X, min(k_max, len(rows) - 1), seed=seed, standardize=standardize)
            k = picked["k"]

        if k < 2:
            raise ValueError("k must be >= 2")
        if k > len(rows):
            raise ValueError(f"k ({k}) cannot exceed number of rows ({len(rows)})")

        model = KMeans(n_clusters=k, seed=seed, standardize=standardize).fit(X, feature_names=features)
        sil = silhouette_score(X, model.labels_) if k < len(X) else None
        pca = PCA(n_components=2).fit(X)
        Z = pca.transform(X)
        curve = elbow(X, min(k_max, len(rows) - 1), seed=seed, standardize=standardize)

        result = {
            "ok": True,
            "overview": {"rows": len(rows), "features": features, "k": k,
                         "standardized": standardize, "auto_k": auto_k},
            "inertia": round(model.inertia_, 4),
            "silhouette": round(sil, 4) if sil is not None else None,
            "sizes": model.sizes(),
            "centroids": [[round(float(v), 4) for v in c] for c in model.centroids_],
            "profile": cluster_profile(X, model.labels_, features),
            "pca": {
                "x": [round(float(v), 4) for v in Z[:, 0]],
                "y": [round(float(v), 4) for v in Z[:, 1]],
                "cluster": [int(c) for c in model.labels_],
                "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            },
            "elbow": curve,
        }
        return jsonify(result)
    except Exception as err:
        return jsonify({"ok": False, "error": str(err)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
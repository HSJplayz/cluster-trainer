"""From-scratch unsupervised learning: k-means (k-means++ init, Lloyd's
algorithm, multiple restarts), PCA via SVD, the elbow method, and silhouette
scores. Depends only on numpy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 1e-12
DEFAULT_SEED = 42


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < EPS] = 1.0
    return (X - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


def _sq_dists(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Pairwise squared distances (n, k) without materializing n*k*d tensors."""
    x2 = (X**2).sum(axis=1)[:, None]
    c2 = (centers**2).sum(axis=1)[None, :]
    d2 = x2 + c2 - 2 * X @ centers.T
    return np.maximum(d2, 0.0)


def _kmeans_plusplus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    closest = _sq_dists(X, np.asarray(centers))[:, 0]
    for _ in range(1, k):
        probs = closest / closest.sum() if closest.sum() > EPS else np.full(n, 1.0 / n)
        idx = rng.choice(n, p=probs)
        centers.append(X[idx])
        d = _sq_dists(X, X[idx : idx + 1])[:, 0]
        closest = np.minimum(closest, d)
    return np.asarray(centers)


class KMeans:
    """k-means with k-means++ seeding and multiple restarts (best inertia wins).
    Features are optionally standardized; centroids are kept in the original
    space for interpretability and in scaled space for assignment."""

    def __init__(self, n_clusters: int = 3, *, n_init: int = 10, max_iters: int = 300,
                 tol: float = 1e-6, seed: int = DEFAULT_SEED, standardize: bool = True):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.standardize = standardize
        self.centroids_: np.ndarray | None = None       # original space
        self._centroids_scaled: np.ndarray | None = None
        self.scaler: dict | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0
        self.feature_names: list[str] | None = None

    def _fit_once(self, Xs: np.ndarray, rng: np.random.Generator):
        centers = _kmeans_plusplus(Xs, self.n_clusters, rng)
        labels = np.zeros(Xs.shape[0], dtype=int)
        for it in range(self.max_iters):
            d2 = _sq_dists(Xs, centers)
            new_labels = d2.argmin(axis=1)
            if np.array_equal(new_labels, labels) and it > 0:
                labels = new_labels
                break
            labels = new_labels
            for c in range(self.n_clusters):
                pts = Xs[labels == c]
                if len(pts):
                    centers[c] = pts.mean(axis=0)
        inertia = float(_sq_dists(Xs, centers).min(axis=1).sum())
        return centers, labels, inertia, it + 1

    def fit(self, X, *, feature_names=None) -> "KMeans":
        X = _as_2d(X)
        n = X.shape[0]
        if self.n_clusters > n:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot exceed n_samples ({n})")
        self.feature_names = list(feature_names) if feature_names else None
        if self.standardize:
            Xs, self.scaler = _standardize(X)
        else:
            Xs, self.scaler = X, None

        best = None
        for i in range(self.n_init):
            rng = np.random.default_rng(self.seed + i)
            centers, labels, inertia, iters = self._fit_once(Xs, rng)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, iters)
        centers, labels, inertia, iters = best
        self._centroids_scaled = centers
        self.centroids_ = self._to_original(centers)
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iters
        return self

    def _to_original(self, centers: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return centers
        return centers * np.asarray(self.scaler["std"]) + np.asarray(self.scaler["mean"])

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        return (X - np.asarray(self.scaler["mean"])) / np.asarray(self.scaler["std"])

    def predict(self, X) -> np.ndarray:
        if self._centroids_scaled is None:
            raise RuntimeError("model has not been fitted")
        Xs = self._scale(_as_2d(X))
        return _sq_dists(Xs, self._centroids_scaled).argmin(axis=1)

    def transform(self, X) -> np.ndarray:
        """Distance from each sample to each centroid (original units)."""
        if self.centroids_ is None:
            raise RuntimeError("model has not been fitted")
        return np.sqrt(_sq_dists(_as_2d(X), self.centroids_))

    def sizes(self) -> list[int]:
        return np.bincount(self.labels_, minlength=self.n_clusters).tolist()

    def params(self) -> dict:
        return {
            "_model": "KMeans",
            "n_clusters": self.n_clusters,
            "centroids": self.centroids_.tolist(),
            "scaler": self.scaler,
            "inertia": self.inertia_,
            "n_features": int(self.centroids_.shape[1]),
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_params(cls, p: dict) -> "KMeans":
        model = cls(n_clusters=p["n_clusters"], standardize=p.get("scaler") is not None)
        model.centroids_ = np.asarray(p["centroids"], dtype=float)
        model.scaler = p.get("scaler")
        model._centroids_scaled = (
            (model.centroids_ - np.asarray(model.scaler["mean"])) / np.asarray(model.scaler["std"])
            if model.scaler else model.centroids_
        )
        model.inertia_ = p.get("inertia")
        model.feature_names = p.get("feature_names")
        return model


class PCA:
    """Principal component analysis via SVD of the centered data matrix."""

    def __init__(self, n_components: int = 2):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def fit(self, X) -> "PCA":
        X = _as_2d(X)
        n = X.shape[0]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # economy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]
        var = (S**2) / max(n - 1, 1)
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum() if var.sum() > EPS else var[:k]
        self.singular_values_ = S[:k]
        return self

    def transform(self, X) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("model has not been fitted")
        return (_as_2d(X) - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def params(self) -> dict:
        return {
            "_model": "PCA",
            "mean": self.mean_.tolist(),
            "components": self.components_.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
        }

    @classmethod
    def from_params(cls, p: dict) -> "PCA":
        model = cls(n_components=len(p["components"]))
        model.mean_ = np.asarray(p["mean"], dtype=float)
        model.components_ = np.asarray(p["components"], dtype=float)
        model.explained_variance_ratio_ = np.asarray(p["explained_variance_ratio"], dtype=float)
        return model


def silhouette_score(X, labels) -> float:
    """Mean silhouette coefficient in [-1, 1] (O(n^2), for small datasets)."""
    X = _as_2d(X)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= X.shape[0]:
        raise ValueError("silhouette needs 2 <= n_clusters < n_samples")
    # full pairwise distance matrix
    d = np.sqrt(np.maximum(
        (X**2).sum(axis=1)[:, None] + (X**2).sum(axis=1)[None, :] - 2 * X @ X.T, 0.0))
    scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        own = labels == labels[i]
        own[i] = False
        if own.sum() == 0:
            scores[i] = 0.0
            continue
        a = d[i, own].mean()
        b = min(d[i, labels == c].mean() for c in uniq if c != labels[i])
        scores[i] = (b - a) / max(a, b) if max(a, b) > EPS else 0.0
    return float(scores.mean())


def elbow(X, k_max: int = 10, *, seed: int = DEFAULT_SEED, standardize: bool = True,
          with_silhouette: bool = True) -> dict:
    """Inertia (and optional silhouette) for k = 1..k_max."""
    X = _as_2d(X)
    ks, inertias, silhouettes = [], [], []
    for k in range(1, k_max + 1):
        if k > X.shape[0]:
            break
        model = KMeans(n_clusters=k, seed=seed, standardize=standardize).fit(X)
        ks.append(k)
        inertias.append(round(model.inertia_, 4))
        if with_silhouette and k >= 2:
            silhouettes.append(round(silhouette_score(X, model.labels_), 4))
        elif with_silhouette:
            silhouettes.append(None)
    return {"k": ks, "inertia": inertias, "silhouette": silhouettes}


def best_k_by_silhouette(X, k_max: int = 10, *, seed: int = DEFAULT_SEED,
                         standardize: bool = True) -> dict:
    curve = elbow(X, k_max, seed=seed, standardize=standardize, with_silhouette=True)
    valid = [(s, k) for k, s in zip(curve["k"], curve["silhouette"]) if s is not None]
    if not valid:
        raise ValueError("could not evaluate any k")
    score, k = max(valid)
    return {"k": int(k), "silhouette": score, "curve": curve}


def cluster_profile(X, labels, feature_names=None) -> list[dict]:
    X = _as_2d(X)
    labels = np.asarray(labels)
    names = list(feature_names) if feature_names else [f"f{i}" for i in range(X.shape[1])]
    profiles = []
    for c in np.unique(labels):
        pts = X[labels == c]
        profiles.append({
            "cluster": int(c),
            "size": int(pts.shape[0]),
            "means": {n: round(float(v), 4) for n, v in zip(names, pts.mean(axis=0))},
            "std": {n: round(float(v), 4) for n, v in zip(names, pts.std(axis=0))},
        })
    return profiles


# --- persistence --------------------------------------------------------------

_REGISTRY = {"KMeans": KMeans, "PCA": PCA}


def save_model(model, path) -> str:
    Path(path).write_text(json.dumps(model.params(), indent=2), encoding="utf-8")
    return str(path)


def load_model(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tag = data.get("_model")
    if tag not in _REGISTRY:
        raise ValueError(f"unknown model type: {tag!r}")
    return _REGISTRY[tag].from_params(data)
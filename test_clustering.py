"""Tests for cluster-trainer: k-means, PCA, metrics, persistence, CLI, Flask."""

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from clustering import (
    KMeans,
    PCA,
    best_k_by_silhouette,
    cluster_profile,
    elbow,
    load_model,
    save_model,
    silhouette_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def blobs(centers, per=40, spread=0.4, seed=1):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, spread, size=(per, len(c))) for c in centers])
    y = np.repeat(np.arange(len(centers)), per)
    return X, y


def match_clusters(pred, truth):
    """Fraction of points whose predicted cluster maps to the correct true label."""
    mapping = {}
    for p, t in zip(pred, truth):
        counts = mapping.setdefault(p, {})
        counts[t] = counts.get(t, 0) + 1
    mapping = {p: max(c, key=c.get) for p, c in mapping.items()}
    return np.mean([mapping[p] == t for p, t in zip(pred, truth)])


class KMeansTest(unittest.TestCase):
    def test_recovers_separated_blobs(self):
        X, y = blobs([(0, 0), (6, 6), (12, 0)], seed=3)
        m = KMeans(n_clusters=3, seed=1).fit(X)
        self.assertGreater(match_clusters(m.labels_, y), 0.99)

    def test_inertia_decreases_with_k(self):
        X, _ = blobs([(0, 0), (5, 5)], per=50, seed=4)
        inertias = [KMeans(n_clusters=k, seed=1).fit(X).inertia_ for k in (1, 2, 3)]
        self.assertGreater(inertias[0], inertias[1])
        self.assertGreater(inertias[1], inertias[2])

    def test_reproducible_with_seed(self):
        X, _ = blobs([(0, 0), (5, 5)], per=30, seed=5)
        a = KMeans(n_clusters=2, seed=99).fit(X)
        b = KMeans(n_clusters=2, seed=99).fit(X)
        np.testing.assert_allclose(a.centroids_, b.centroids_)

    def test_too_many_clusters_raises(self):
        X, _ = blobs([(0, 0)], per=3)
        with self.assertRaises(ValueError):
            KMeans(n_clusters=10).fit(X)

    def test_predict_matches_training_labels(self):
        X, _ = blobs([(0, 0), (6, 6)], per=25, seed=6)
        m = KMeans(n_clusters=2, seed=2).fit(X)
        np.testing.assert_array_equal(m.predict(X), m.labels_)

    def test_predict_new_points_near_centroid(self):
        X, _ = blobs([(0, 0), (10, 10)], per=40, seed=7)
        m = KMeans(n_clusters=2, seed=2).fit(X)
        pred = m.predict([[0.1, 0.1], [10.2, 9.8]])
        self.assertNotEqual(pred[0], pred[1])

    def test_transform_shape(self):
        X, _ = blobs([(0, 0), (5, 5)], per=10, seed=8)
        m = KMeans(n_clusters=2, seed=1).fit(X)
        self.assertEqual(m.transform(X).shape, (20, 2))

    def test_sizes_sum_to_n(self):
        X, _ = blobs([(0, 0), (5, 5), (0, 5)], per=15, seed=9)
        m = KMeans(n_clusters=3, seed=1).fit(X)
        self.assertEqual(sum(m.sizes()), 45)

    def test_no_standardize_path(self):
        X, _ = blobs([(0, 0), (5, 5)], per=20, seed=10)
        m = KMeans(n_clusters=2, standardize=False, seed=1).fit(X)
        self.assertIsNone(m.scaler)
        self.assertIsNotNone(m.centroids_)

    def test_standardize_populates_scaler(self):
        X, _ = blobs([(0, 0), (5, 5)], per=20, seed=11)
        m = KMeans(n_clusters=2, standardize=True, seed=1).fit(X)
        self.assertIn("mean", m.scaler)
        self.assertIn("std", m.scaler)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            KMeans(n_clusters=2).predict([[1, 2]])


class PCATest(unittest.TestCase):
    def test_variance_descending(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 4))
        p = PCA(n_components=3).fit(X)
        self.assertTrue(np.all(np.diff(p.explained_variance_) <= 1e-9))

    def test_captures_dominant_direction(self):
        rng = np.random.default_rng(2)
        t = rng.normal(size=200)
        X = np.column_stack([t * 5, t * 5 + rng.normal(scale=0.05, size=200)])
        p = PCA(n_components=2).fit(X)
        self.assertGreater(p.explained_variance_ratio_[0], 0.99)

    def test_transform_shape_and_consistency(self):
        X = np.random.default_rng(3).normal(size=(30, 5))
        p = PCA(n_components=2).fit(X)
        self.assertEqual(p.transform(X).shape, (30, 2))
        np.testing.assert_allclose(p.fit_transform(X), p.transform(X))

    def test_components_orthonormal(self):
        X = np.random.default_rng(4).normal(size=(50, 4))
        p = PCA(n_components=2).fit(X)
        gram = p.components_ @ p.components_.T
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-9)

    def test_full_ratio_sums_to_one(self):
        X = np.random.default_rng(5).normal(size=(40, 3))
        p = PCA(n_components=3).fit(X)
        self.assertAlmostEqual(float(p.explained_variance_ratio_.sum()), 1.0, places=9)

    def test_two_components_on_1d_data(self):
        X = np.random.default_rng(6).normal(size=(20, 1))
        p = PCA(n_components=2).fit(X)
        self.assertEqual(p.components_.shape[0], 1)


class MetricsTest(unittest.TestCase):
    def test_silhouette_high_when_separated(self):
        X, y = blobs([(0, 0), (10, 10)], per=30, spread=0.3, seed=12)
        self.assertGreater(silhouette_score(X, y), 0.8)

    def test_silhouette_low_when_overlapping(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(60, 2))
        y = np.array([0] * 30 + [1] * 30)
        self.assertLess(silhouette_score(X, y), 0.3)

    def test_silhouette_single_cluster_raises(self):
        X = np.random.default_rng(14).normal(size=(10, 2))
        with self.assertRaises(ValueError):
            silhouette_score(X, np.zeros(10, dtype=int))

    def test_elbow_structure(self):
        X, _ = blobs([(0, 0), (5, 5), (0, 5)], per=20, seed=15)
        c = elbow(X, 5)
        self.assertEqual(c["k"], [1, 2, 3, 4, 5])
        inertias = c["inertia"]
        self.assertTrue(all(inertias[i] > inertias[i + 1] for i in range(len(inertias) - 1)))
        self.assertIsNone(c["silhouette"][0])

    def test_best_k_picks_true_count(self):
        X, _ = blobs([(0, 0), (8, 0), (4, 7)], per=40, seed=16)
        picked = best_k_by_silhouette(X, 6)
        self.assertEqual(picked["k"], 3)

    def test_cluster_profile_means(self):
        X = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 10.0], [10.0, 12.0]])
        labels = np.array([0, 0, 1, 1])
        prof = cluster_profile(X, labels, ["a", "b"])
        self.assertEqual(prof[0]["size"], 2)
        self.assertAlmostEqual(prof[0]["means"]["a"], 0.0)
        self.assertAlmostEqual(prof[1]["means"]["b"], 11.0)


class PersistenceTest(unittest.TestCase):
    def test_kmeans_round_trip(self):
        X, _ = blobs([(0, 0), (6, 6)], per=25, seed=17)
        m = KMeans(n_clusters=2, seed=3).fit(X, feature_names=["x", "y"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "km.json")
            save_model(m, path)
            loaded = load_model(path)
        np.testing.assert_allclose(loaded.centroids_, m.centroids_)
        np.testing.assert_array_equal(loaded.predict(X), m.predict(X))
        self.assertEqual(loaded.feature_names, ["x", "y"])

    def test_pca_round_trip(self):
        X = np.random.default_rng(18).normal(size=(30, 4))
        p = PCA(n_components=2).fit(X)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pca.json")
            save_model(p, path)
            loaded = load_model(path)
        np.testing.assert_allclose(loaded.components_, p.components_)
        np.testing.assert_allclose(loaded.transform(X), p.transform(X))

    def test_unknown_model_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"_model": "Nope"}, f)
            with self.assertRaises(ValueError):
                load_model(path)


class _Cli:
    def cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(HERE, "cli.py"), *args],
                              capture_output=True, text=True, cwd=HERE)


class CliTest(unittest.TestCase, _Cli):
    def _csv(self, d):
        path = os.path.join(d, "data.csv")
        rng = np.random.default_rng(20)
        rows = []
        for c in [(0, 0), (6, 6), (12, 0)]:
            for _ in range(20):
                rows.append([round(random_val(c[0], rng), 3), round(random_val(c[1], rng), 3)])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["x", "y"])
            w.writerows(rows)
        return path

    def test_cluster_writes_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._csv(d)
            labels = os.path.join(d, "labels.csv")
            model = os.path.join(d, "model.json")
            r = self.cli("cluster", src, "-k", "3", "--labels", labels, "--save", model)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("k-means", r.stdout)
            with open(labels, encoding="utf-8") as f:
                self.assertEqual(next(csv.reader(f)), ["x", "y", "cluster"])
            self.assertEqual(load_model(model).n_clusters, 3)

    def test_elbow_output(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.cli("elbow", self._csv(d), "--k-max", "4")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("silhouette", r.stdout)

    def test_pca_writes_csv(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "pca.csv")
            r = self.cli("pca", self._csv(d), "-o", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(next(csv.reader(f)), ["pc1", "pc2"])

    def test_demo_ok(self):
        r = self.cli("demo", "-k", "3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Silhouette", r.stdout)

    def test_missing_column_returns_1(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.cli("cluster", self._csv(d), "-k", "3", "-f", "nope")
            self.assertEqual(r.returncode, 1)
            self.assertIn("error:", r.stderr)

    def test_missing_file_returns_1(self):
        r = self.cli("cluster", "does-not-exist.csv", "-k", "2")
        self.assertEqual(r.returncode, 1)


def random_val(c, rng):
    return c + rng.normal(scale=0.5)


class FlaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as webapp
        webapp.app.config["TESTING"] = True
        cls.client = webapp.app.test_client()

    def _csv_text(self):
        rng = np.random.default_rng(21)
        lines = ["x,y"]
        for c in [(0, 0), (6, 6), (12, 0)]:
            for _ in range(15):
                lines.append(f"{c[0]+rng.normal(scale=0.5):.3f},{c[1]+rng.normal(scale=0.5):.3f}")
        return "\n".join(lines)

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json(), {"ok": True})

    def test_index_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"cluster-trainer", r.data)

    def test_cluster_endpoint(self):
        r = self.client.post("/api/cluster", json={"csv": self._csv_text(), "features": ["x", "y"], "k": 3})
        data = r.get_json()
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["overview"]["k"], 3)
        self.assertEqual(len(data["pca"]["x"]), 45)
        self.assertEqual(len(data["profile"]), 3)
        self.assertEqual(data["elbow"]["k"][-1], 8)

    def test_auto_k(self):
        r = self.client.post("/api/cluster", json={"csv": self._csv_text(), "features": ["x", "y"], "auto_k": True, "k_max": 6})
        data = r.get_json()
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["overview"]["k"], 3)

    def test_bad_feature_returns_400(self):
        r = self.client.post("/api/cluster", json={"csv": self._csv_text(), "features": ["nope"], "k": 3})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(r.get_json()["ok"])

    def test_empty_csv_returns_400(self):
        r = self.client.post("/api/cluster", json={"csv": "", "features": [], "k": 3})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)

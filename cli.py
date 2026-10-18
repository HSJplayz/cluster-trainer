"""CLI for cluster-trainer: cluster, elbow, pca, demo."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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


def _read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"'{path}' has no data rows")
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


def _extract(rows, features):
    return np.asarray([[float(r[f]) for f in features] for r in rows], dtype=float)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cluster-trainer",
        description="k-means clustering and PCA from scratch (numpy only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py cluster customers.csv -k 4\n"
            "  python cli.py cluster customers.csv -k 4 --labels labelled.csv --save model.json\n"
            "  python cli.py cluster customers.csv --auto-k --k 10\n"
            "  python cli.py elbow customers.csv --k-max 10\n"
            "  python cli.py pca customers.csv -o pca2d.csv\n"
            "  python cli.py demo -k 3"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cluster", help="fit k-means and report clusters")
    c.add_argument("input")
    c.add_argument("-f", "--features", action="append", help="feature column(s); default: all numeric")
    c.add_argument("-k", "--k", type=int, help="number of clusters")
    c.add_argument("--auto-k", action="store_true", help="pick k by best silhouette (up to --k)")
    c.add_argument("--no-standardize", action="store_true", help="do not standardize features")
    c.add_argument("--labels", help="write feature columns + 'cluster' to this CSV")
    c.add_argument("--save", help="save the fitted model to this JSON file")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("-q", "--quiet", action="store_true")

    e = sub.add_parser("elbow", help="inertia + silhouette for k = 1..k_max")
    e.add_argument("input")
    e.add_argument("-f", "--features", action="append")
    e.add_argument("--k-max", type=int, default=10)
    e.add_argument("--no-standardize", action="store_true")
    e.add_argument("--seed", type=int, default=42)

    pc = sub.add_parser("pca", help="project data onto its top principal components")
    pc.add_argument("input")
    pc.add_argument("-f", "--features", action="append")
    pc.add_argument("-n", "--components", type=int, default=2)
    pc.add_argument("-o", "--output", help="write projections to a CSV (default: stdout)")
    pc.add_argument("--save", help="save the PCA model JSON")

    d = sub.add_parser("demo", help="cluster synthetic 2-D blobs")
    d.add_argument("-k", "--k", type=int, default=4)
    d.add_argument("--points", type=int, default=300)
    d.add_argument("--noise", type=float, default=0.6)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("-q", "--quiet", action="store_true")

    return p


def cmd_cluster(args, rows):
    features = args.features or _numeric_cols(rows)
    if not features:
        raise ValueError("no numeric feature columns found (use -f)")
    X = _extract(rows, features)
    standardize = not args.no_standardize

    if args.auto_k:
        k_max = args.k or 10
        picked = best_k_by_silhouette(X, k_max, seed=args.seed, standardize=standardize)
        k = picked["k"]
        if not args.quiet:
            print(f"Auto-k: chose k={k} (silhouette={picked['silhouette']:.4f})")
    else:
        k = args.k or 3

    model = KMeans(n_clusters=k, seed=args.seed, standardize=standardize).fit(X, feature_names=features)
    sil = None
    if k >= 2 and k < len(X):
        sil = silhouette_score(X, model.labels_)

    if not args.quiet:
        print(f"k-means: {len(rows)} rows, {len(features)} feature(s), k={k}, "
              f"standardized={standardize}")
        print(f"Inertia: {model.inertia_:.3f}   Iterations: {model.n_iter_}"
              + (f"   Silhouette: {sil:.4f}" if sil is not None else ""))
        print(f"Cluster sizes: {model.sizes()}")
        print("Cluster centers (original units):")
        for c, cen in enumerate(model.centroids_):
            vals = ", ".join(f"{n}={v:.3f}" for n, v in zip(features, cen))
            print(f"  cluster {c}: {vals}")

    if args.labels:
        with open(args.labels, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(features) + ["cluster"])
            for r, c in zip(rows, model.labels_):
                w.writerow([r[f] for f in features] + [int(c)])
        if not args.quiet:
            print(f"Wrote cluster labels -> {args.labels}")
    if args.save:
        save_model(model, args.save)
        if not args.quiet:
            print(f"Saved model -> {args.save}")
    return 0


def cmd_elbow(args, rows):
    features = args.features or _numeric_cols(rows)
    X = _extract(rows, features)
    curve = elbow(X, args.k_max, seed=args.seed, standardize=not args.no_standardize)
    print(f"{'k':>3}{'inertia':>14}{'silhouette':>14}")
    for k, inertia, sil in zip(curve["k"], curve["inertia"], curve["silhouette"]):
        print(f"{k:>3}{inertia:>14.3f}{(f'{sil:.4f}' if sil is not None else '-'):>14}")
    return 0


def cmd_pca(args, rows):
    features = args.features or _numeric_cols(rows)
    X = _extract(rows, features)
    n = min(args.components, len(features))
    model = PCA(n_components=n).fit(X)
    Z = model.transform(X)
    if not args.output:
        print(f"Explained variance ratio: {[round(float(v), 4) for v in model.explained_variance_ratio_]}")
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([f"pc{i + 1}" for i in range(n)])
            for row in Z:
                w.writerow([round(float(v), 6) for v in row])
        print(f"Wrote {len(Z)} projections -> {args.output}")
    if args.save:
        save_model(model, args.save)
        print(f"Saved PCA model -> {args.save}")
    return 0


def cmd_demo(args):
    rng = np.random.default_rng(args.seed)
    centers = [(0, 0), (5, 1), (1, 5), (6, 6)][: args.k]
    per = max(1, args.points // len(centers))
    X = np.vstack([rng.normal(c, args.noise, size=(per, 2)) for c in centers])
    model = KMeans(n_clusters=args.k, seed=args.seed).fit(X, feature_names=["x1", "x2"])
    sil = silhouette_score(X, model.labels_) if args.k < len(X) else None
    if not args.quiet:
        print(f"Demo: {len(X)} points, {len(centers)} true blobs, k={args.k}")
        print(f"Inertia: {model.inertia_:.3f}" + (f"   Silhouette: {sil:.4f}" if sil is not None else ""))
        print(f"Cluster sizes: {model.sizes()}")
        for c, cen in enumerate(model.centroids_):
            print(f"  cluster {c} center: ({cen[0]:.3f}, {cen[1]:.3f})  (true {centers[c]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cluster":
            return cmd_cluster(args, _read_csv(args.input))
        if args.command == "elbow":
            return cmd_elbow(args, _read_csv(args.input))
        if args.command == "pca":
            return cmd_pca(args, _read_csv(args.input))
        if args.command == "demo":
            return cmd_demo(args)
    except (ValueError, OSError, KeyError, IndexError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except Exception as err:  # pragma: no cover - safety net
        print(f"error: unexpected failure: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
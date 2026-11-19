from typing import Iterable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(n_clusters: int, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=10,
                ),
            ),
        ]
    )


def fit_kmeans(
    feature_matrix: pd.DataFrame, n_clusters: int
) -> tuple[Pipeline, pd.Series]:
    if n_clusters < 2:
        raise ValueError("K-Means requires at least 2 clusters.")
    if n_clusters >= len(feature_matrix):
        raise ValueError("Number of clusters must be less than the customer count.")

    pipeline = build_pipeline(n_clusters)
    labels = pipeline.fit_predict(feature_matrix)
    return pipeline, pd.Series(labels, index=feature_matrix.index, name="cluster")


def evaluate_k_values(
    feature_matrix: pd.DataFrame,
    k_values: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        if k < 2 or k >= len(feature_matrix):
            continue
        pipeline = build_pipeline(k, random_state=random_state)
        labels = pipeline.fit_predict(feature_matrix)
        rows.append(
            {
                "k": k,
                "inertia": pipeline.named_steps["kmeans"].inertia_,
                "silhouette": silhouette_score(
                    pipeline.named_steps["scaler"].transform(feature_matrix),
                    labels,
                    sample_size=min(2000, len(feature_matrix)),
                    random_state=random_state,
                ),
            }
        )
    return pd.DataFrame(rows)


def pca_visualization(
    feature_matrix: pd.DataFrame, pipeline: Pipeline
) -> pd.DataFrame:
    scaled = pipeline.named_steps["scaler"].transform(feature_matrix)
    coordinates = PCA(n_components=2, random_state=42).fit_transform(scaled)
    return pd.DataFrame(
        coordinates,
        index=feature_matrix.index,
        columns=["PCA 1", "PCA 2"],
    )

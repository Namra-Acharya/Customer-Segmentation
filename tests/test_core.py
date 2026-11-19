import pandas as pd
from unittest.mock import patch

from src.data_processing import prepare_feature_matrix
from src.modeling import evaluate_k_values, fit_kmeans, pca_visualization
from src.profiling import cluster_profile, create_segment_profiles


def test_prepare_feature_matrix_imputes_missing_values():
    df = pd.DataFrame(
        {"age": [20, None, 40], "annual_spend_inr": [100, 200, None]}
    )
    original, prepared = prepare_feature_matrix(
        df, ["age", "annual_spend_inr"]
    )
    assert original.isna().sum().sum() == 2
    assert prepared.isna().sum().sum() == 0


def test_kmeans_evaluation_and_pca():
    df = pd.DataFrame(
        {
            "age": [20, 21, 50, 52, 30, 31],
            "annual_spend_inr": [10, 12, 90, 95, 40, 42],
        }
    )
    _, prepared = prepare_feature_matrix(
        df, ["age", "annual_spend_inr"]
    )
    pipeline, labels = fit_kmeans(prepared, 3)
    evaluation = evaluate_k_values(prepared, [2, 3])
    coordinates = pca_visualization(prepared, pipeline)

    assert labels.nunique() == 3
    assert set(evaluation.columns) == {"k", "inertia", "silhouette"}
    assert coordinates.shape == (6, 2)


def test_silhouette_evaluation_samples_large_datasets():
    df = pd.DataFrame(
        {
            "feature": list(range(1250)) + list(range(1250, 2500)),
            "other": [0] * 1250 + [10] * 1250,
        }
    )
    _, prepared = prepare_feature_matrix(df, ["feature", "other"])

    with patch("src.modeling.silhouette_score", return_value=0.5) as score:
        evaluate_k_values(prepared, [2])

    assert score.call_args.kwargs["sample_size"] == 2000
    assert score.call_args.kwargs["random_state"] == 42


def test_profiles_and_segment_names():
    df = pd.DataFrame(
        {
            "annual_spend_inr": [100, 1000],
            "annual_income_inr": [200, 2000],
            "annual_transactions": [2, 12],
            "monthly_website_visits": [1, 8],
            "discount_usage_rate": [0.8, 0.1],
            "days_since_last_purchase": [90, 5],
            "loyalty_years": [1, 8],
        }
    )
    labels = pd.Series([0, 1], name="cluster")
    profile = cluster_profile(df, labels, df.columns.tolist())
    segment = create_segment_profiles(profile)

    assert len(segment) == 2
    assert set(segment["segment"]) <= {
        "High-value customers",
        "Budget-conscious customers",
        "Low-engagement customers",
        "Loyal customers",
        "Established customers",
        "Balanced customers",
    }


def test_profiles_support_selected_feature_subset():
    profile = pd.DataFrame(
        {
            "cluster": [0, 1],
            "customer_count": [3, 3],
            "return_rate": [0.05, 0.10],
            "loyalty_years": [2.0, 8.0],
        }
    )
    segment = create_segment_profiles(profile)

    assert len(segment) == 2
    assert segment["segment"].notna().all()

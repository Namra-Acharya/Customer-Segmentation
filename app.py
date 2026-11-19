import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import (
    load_csv,
    numeric_feature_candidates,
    prepare_feature_matrix,
)
from src.modeling import evaluate_k_values, fit_kmeans, pca_visualization
from src.profiling import cluster_profile, create_segment_profiles


st.set_page_config(
    page_title="ClusterIQ",
    page_icon="CI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_default_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #667085;
        --blue: #2563eb;
        --blue-soft: #eff6ff;
        --line: #e7ecf3;
        --surface: #ffffff;
        --canvas: #f6f8fb;
    }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(246,248,251,.92); }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
        padding: 0.7rem 1rem;
    }
    [data-testid="stSidebar"] h1 { font-size: 1.55rem; margin-bottom: 0; }
    [data-testid="stMainBlockContainer"] { max-width: 1250px; padding-top: 1.4rem; }
    h1, h2, h3, p, label, span { color: var(--ink); }
    .block-container { padding-bottom: 3rem; }
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 5px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border: none;
        border-radius: 8px;
        color: var(--muted);
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: var(--blue-soft);
        color: var(--blue);
    }
    .stButton > button, .stDownloadButton > button {
        min-height: 46px;
        border-radius: 9px;
        border: 1px solid #d9e2ef;
        font-weight: 750;
    }
    .stButton > button[kind="primary"] {
        background: var(--blue);
        border-color: var(--blue);
        color: #ffffff;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div { color: var(--blue); }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px;
    }
    .metric-card {
        background: #ffffff; border: 1px solid #e8edf3; border-radius: 14px;
        padding: 18px 20px; box-shadow: 0 3px 12px rgba(15,23,42,.04);
        min-height: 112px;
    }
    .metric-label { color: #64748b; font-size: .82rem; font-weight: 700; text-transform: uppercase; }
    .metric-value { color: #0f172a; font-size: 1.65rem; font-weight: 750; margin-top: 8px; }
    h1, h2, h3 { color: #0f172a; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "customer_segmentation_12000.csv"

with st.sidebar:
    st.title("ClusterIQ")
    st.caption("Customer Intelligence & Segmentation")
    st.divider()
    st.markdown("**Data source**")
    uploaded = st.file_uploader("Upload a customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
            st.session_state["data_source"] = "Uploaded CSV"
            if st.session_state.get("uploaded_name") != uploaded.name:
                for key in (
                    "cluster_labels",
                    "n_clusters",
                    "selected_features",
                    "pipeline",
                    "coordinates",
                    "assigned_customers",
                    "profile_df",
                ):
                    st.session_state.pop(key, None)
                st.session_state["uploaded_name"] = uploaded.name
        except Exception as exc:
            st.error(str(exc))
            df = load_default_data(str(DEFAULT_CSV))
            st.session_state["data_source"] = "Default dataset"
    else:
        df = load_default_data(str(DEFAULT_CSV))
        st.session_state["data_source"] = "Default dataset"
    st.success(st.session_state["data_source"])


tabs = st.tabs(
    ["Overview", "Data Explorer", "Segmentation", "Customer Insights", "Download"]
)

numeric_features = numeric_feature_candidates(df)

with tabs[0]:
    st.title("Customer overview")
    st.caption("A data-driven view of the loaded customer base")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total customers", f"{len(df):,}")
    with c2:
        clusters = st.session_state.get("n_clusters")
        metric_card("Discovered clusters", str(clusters) if clusters else "Not run")
    with c3:
        avg_spend = df["annual_spend_inr"].mean() if "annual_spend_inr" in df else None
        metric_card(
            "Average annual spend",
            f"₹{avg_spend:,.0f}" if pd.notna(avg_spend) else "Unavailable",
        )
    with c4:
        avg_income = (
            df["annual_income_inr"].mean()
            if "annual_income_inr" in df
            else None
        )
        metric_card(
            "Average annual income",
            f"₹{avg_income:,.0f}" if pd.notna(avg_income) else "Unavailable",
        )

    st.subheader("Cluster distribution")
    labels = st.session_state.get("cluster_labels")
    if labels is None:
        st.info("Run K-Means in the Segmentation tab to populate this view.")
    else:
        dist = labels.value_counts().sort_index().rename_axis("Cluster").reset_index(
            name="Customers"
        )
        fig = px.bar(
            dist,
            x="Cluster",
            y="Customers",
            color="Cluster",
            text="Customers",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.title("Data explorer")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Dataset preview")
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Dataset health")
        missing = df.isna().sum()
        health = pd.DataFrame(
            {"Column": missing.index, "Missing values": missing.values}
        )
        st.dataframe(
            health[health["Missing values"] > 0],
            use_container_width=True,
            hide_index=True,
        )
        if missing.sum() == 0:
            st.success("No missing values detected")

    st.subheader("Numerical feature statistics")
    st.dataframe(
        df[numeric_features].describe().T.round(2),
        use_container_width=True,
    )

with tabs[2]:
    st.title("Customer segmentation")
    st.caption("Scale selected features, run K-Means, then inspect a PCA overview.")

    left, right = st.columns([1, 3])
    with left:
        st.markdown("### Model setup")
        selected_features = st.multiselect(
            "Clustering features",
            numeric_features,
            default=numeric_features,
            help="customer_id and categorical metadata are excluded.",
        )
        st.caption(f"{len(selected_features)} feature(s) selected")
        n_clusters = st.slider("Number of clusters", 2, 10, 4)
        run_evaluation = st.button(
            "Run K-Means", type="primary", use_container_width=True
        )

    with right:
        try:
            _, prepared = prepare_feature_matrix(df, selected_features)
            with st.spinner("Evaluating 2 to 10 clusters..."):
                evaluation = evaluate_k_values(prepared, range(2, 11))

            st.subheader("Elbow and silhouette evaluation")
            e1, e2 = st.columns(2)
            with e1:
                elbow_fig = px.line(
                    evaluation,
                    x="k",
                    y="inertia",
                    markers=True,
                    labels={"k": "Number of clusters", "inertia": "Inertia"},
                )
                elbow_fig.update_traces(line={"color": "#2563eb", "width": 3})
                elbow_fig.update_layout(height=330, template="plotly_white", margin={"l": 20, "r": 20, "t": 20, "b": 20})
                st.plotly_chart(elbow_fig, use_container_width=True)
            with e2:
                silhouette_fig = px.line(
                    evaluation,
                    x="k",
                    y="silhouette",
                    markers=True,
                    labels={"k": "Number of clusters", "silhouette": "Silhouette score"},
                )
                silhouette_fig.update_traces(line={"color": "#f97316", "width": 3})
                silhouette_fig.update_layout(height=330, template="plotly_white", margin={"l": 20, "r": 20, "t": 20, "b": 20})
                st.plotly_chart(silhouette_fig, use_container_width=True)

            if run_evaluation:
                with st.spinner("Fitting K-Means and preparing cluster outputs..."):
                    pipeline, labels = fit_kmeans(prepared, n_clusters)
                    coordinates = pca_visualization(prepared, pipeline)
                    coordinates["Cluster"] = labels.astype(str)
                    assigned = df.copy()
                    assigned["cluster"] = labels

                    st.session_state["cluster_labels"] = labels
                    st.session_state["n_clusters"] = n_clusters
                    st.session_state["selected_features"] = selected_features
                    st.session_state["pipeline"] = pipeline
                    st.session_state["coordinates"] = coordinates
                    st.session_state["assigned_customers"] = assigned
                    st.session_state["profile_df"] = cluster_profile(
                        df, labels, selected_features
                    )

            coordinates = st.session_state.get("coordinates")
            assigned = st.session_state.get("assigned_customers")
            if coordinates is not None and assigned is not None:
                st.success(
                    f"Model complete: {st.session_state['n_clusters']} clusters "
                    f"for {len(assigned):,} customers."
                )

                st.subheader("Cluster visualization (PCA)")
                viz = px.scatter(
                    coordinates,
                    x="PCA 1",
                    y="PCA 2",
                    color="Cluster",
                    opacity=0.72,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                viz.update_layout(height=480, margin={"l": 20, "r": 20, "t": 20, "b": 20})
                st.plotly_chart(viz, use_container_width=True)

                st.subheader("Assigned customers")
                st.dataframe(
                    assigned.head(100),
                    use_container_width=True,
                    hide_index=True,
                )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"The model could not be run: {exc}")

with tabs[3]:
    st.title("Customer insights")
    profile_df = st.session_state.get("profile_df")
    labels = st.session_state.get("cluster_labels")

    if profile_df is None or labels is None:
        st.info("Run K-Means in the Segmentation tab to generate cluster profiles.")
    else:
        profile_df = profile_df.reset_index(drop=True)
        segment_df = create_segment_profiles(profile_df)
        st.subheader("Data-driven segment profiles")
        segment_map = segment_df.set_index("cluster")["segment"].to_dict()
        cluster_map = labels.map(segment_map)

        cols = st.columns(min(len(segment_df), 4))
        for index, row in segment_df.iterrows():
            with cols[index % len(cols)]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Cluster {row["cluster"]}</div>
                        <div style="font-size:1.15rem;font-weight:750;color:#0f172a;margin:8px 0;">
                            {row["segment"]}
                        </div>
                        <div style="color:#64748b;">{int(row["customer_count"]):,} customers</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("Marketing recommendations")
        st.dataframe(
            segment_df[
                ["cluster", "segment", "customer_count", "recommendation"]
            ].rename(columns={"recommendation": "Suggested action"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These profiles and recommendations are data-driven suggestions, "
            "not guaranteed business outcomes."
        )

        st.subheader("Profile statistics")
        enriched = profile_df.merge(
            segment_df[["cluster", "segment"]],
            on="cluster",
            how="left",
        )
        st.dataframe(
            enriched,
            use_container_width=True,
            hide_index=True,
        )

        if "annual_spend_inr" in enriched:
            spend_fig = px.bar(
                enriched,
                x="segment",
                y="annual_spend_inr",
                color="segment",
                text_auto=".2s",
                labels={"annual_spend_inr": "Average annual spend"},
            )
            spend_fig.update_layout(showlegend=False, height=420)
            st.plotly_chart(spend_fig, use_container_width=True)

        if "annual_income_inr" in enriched:
            income_fig = px.bar(
                enriched,
                x="segment",
                y="annual_income_inr",
                color="segment",
                text_auto=".2s",
                labels={"annual_income_inr": "Average annual income"},
            )
            income_fig.update_layout(showlegend=False, height=420)
            st.plotly_chart(income_fig, use_container_width=True)

with tabs[4]:
    st.title("Download results")
    labels = st.session_state.get("cluster_labels")
    profile_df = st.session_state.get("profile_df")

    if labels is None:
        st.info("Run K-Means before downloading cluster assignments.")
    else:
        assigned = df.copy()
        assigned["cluster"] = labels
        assigned_buffer = io.BytesIO()
        assigned.to_csv(assigned_buffer, index=False)

        st.download_button(
            "Download customer records with clusters",
            data=assigned_buffer.getvalue(),
            file_name="clusteriq_customer_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if profile_df is not None:
            segment_df = create_segment_profiles(profile_df)
            summary_buffer = io.BytesIO()
            segment_df.to_csv(summary_buffer, index=False)
            st.download_button(
                "Download cluster summary",
                data=summary_buffer.getvalue(),
                file_name="clusteriq_cluster_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

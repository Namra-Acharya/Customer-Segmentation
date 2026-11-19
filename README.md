# ClusterIQ - Customer Intelligence & Segmentation

ClusterIQ is an interactive Streamlit application for K-Means customer segmentation. It uploads or loads a customer CSV, scales numerical features, evaluates several cluster counts, visualizes clusters with PCA, and produces data-driven segment profiles and marketing recommendations.

## Dataset

The included `customer_segmentation_12000.csv` contains 12,000 customer records and is provided as a synthetic demonstration dataset. It does not represent actual customers. It contains demographic, purchasing, engagement, and membership-style columns.

No customer ID or categorical metadata is used as an initial clustering feature.

## Features

- CSV upload and dataset preview
- Missing-value and numerical-statistics summaries
- Median imputation for selected features
- StandardScaler plus K-Means pipeline
- Elbow method and silhouette score from K=2 to K=10
- Optional K-Means cluster count selection from 2 to 10
- PCA 2D visualization for inspection only
- Statistics-based cluster profiles
- CSV downloads for assigned customers and cluster summaries

## Run locally

```powershell
cd "<project-folder>"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal. The included dataset is loaded automatically when no CSV is uploaded.

## Project structure

```text
app.py                       Streamlit dashboard
src/data_processing.py       Loading, validation, and missing-value handling
src/modeling.py              Pipeline, K-Means, evaluation, PCA
src/profiling.py             Cluster statistics and segment labels
customer_segmentation_12000.csv
requirements.txt
README.md
```

## Streamlit Community Cloud

1. Create a GitHub repository and push this project.
2. Open `share.streamlit.io`.
3. Choose the repository and set **Main file path** to `app.py`.
4. In **Advanced settings**, choose Python 3.11.
5. Deploy. The app uses a repository-relative dataset path and does not require local file paths.
6. If an older deployment remains stuck, use **Manage app > Reboot app** after the new commit finishes building.

The numerical package versions are pinned in `requirements.txt`.

## Model notes

The initial model uses numerical variables only. Categorical fields such as city tier, channel, and membership can be explored descriptively but are not encoded into version 1. Median imputation changes missing selected values for modeling; original missing values remain visible in the Data Explorer and assigned download unless otherwise transformed.

PCA is used only to project standardized features into two dimensions for visualization. Clustering is fitted in the original selected feature space after scaling.

Segment names are assigned using each cluster's measured average spending, income, transactions, visits, discount usage, recency, and loyalty. Several clusters may receive the same or a fallback label if their statistics do not meet an interpretation rule.

The recommendations are explanatory marketing suggestions, not guaranteed outcomes.

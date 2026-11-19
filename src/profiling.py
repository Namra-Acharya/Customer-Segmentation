import pandas as pd


def cluster_profile(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    assigned = df[feature_names].copy()
    assigned["cluster"] = cluster_labels
    return (
        assigned.groupby("cluster")[feature_names]
        .mean()
        .round(2)
        .assign(customer_count=assigned.groupby("cluster").size())
        .reset_index()
    )


def create_segment_profiles(profile_df: pd.DataFrame) -> pd.DataFrame:
    """Assign business-oriented names from relative cluster statistics."""
    overall_spend = profile_df["annual_spend_inr"].mean() if "annual_spend_inr" in profile_df else float("nan")
    overall_income = profile_df["annual_income_inr"].mean() if "annual_income_inr" in profile_df else float("nan")
    overall_transactions = profile_df["annual_transactions"].mean() if "annual_transactions" in profile_df else float("nan")
    overall_visits = profile_df["monthly_website_visits"].mean() if "monthly_website_visits" in profile_df else float("nan")
    overall_discount = profile_df["discount_usage_rate"].mean() if "discount_usage_rate" in profile_df else float("nan")
    overall_recency = profile_df["days_since_last_purchase"].mean() if "days_since_last_purchase" in profile_df else float("nan")
    overall_loyalty = profile_df["loyalty_years"].mean() if "loyalty_years" in profile_df else float("nan")

    profiles = []
    used_names: set[str] = set()
    for _, row in profile_df.iterrows():
        candidates = []
        if (
            "annual_spend_inr" in profile_df.columns
            and "annual_income_inr" in profile_df.columns
            and
            row["annual_spend_inr"] >= overall_spend
            and row["annual_income_inr"] >= overall_income
        ):
            candidates.append(("High-value customers", 3))
        if (
            "discount_usage_rate" in profile_df.columns
            and "annual_spend_inr" in profile_df.columns
            and
            row["discount_usage_rate"] >= overall_discount
            and row["annual_spend_inr"] < overall_spend
        ):
            candidates.append(("Budget-conscious customers", 2))
        if (
            "monthly_website_visits" in profile_df.columns
            and "annual_transactions" in profile_df.columns
            and "days_since_last_purchase" in profile_df.columns
            and
            row["monthly_website_visits"] <= overall_visits
            and row["annual_transactions"] <= overall_transactions
            and row["days_since_last_purchase"] >= overall_recency
        ):
            candidates.append(("Low-engagement customers", 3))
        if (
            "loyalty_years" in profile_df.columns
            and "annual_transactions" in profile_df.columns
            and
            row["loyalty_years"] >= overall_loyalty
            and row["annual_transactions"] >= overall_transactions
        ):
            candidates.append(("Loyal customers", 2))

        name = "Balanced customers"
        if candidates:
            candidates.sort(key=lambda item: item[1], reverse=True)
            name = candidates[0][0]
            if name in used_names:
                name = "Established customers"
        used_names.add(name)

        profiles.append(
            {
                "cluster": int(row["cluster"]),
                "segment": name,
                "customer_count": int(row["customer_count"]),
                "recommendation": segment_recommendation(
                    name, row, overall_spend, overall_income
                ),
            }
        )
    return pd.DataFrame(profiles)


def segment_recommendation(
    name: str,
    row: pd.Series,
    overall_spend: float,
    overall_income: float,
) -> str:
    recommendations = {
        "High-value customers": (
            "Prioritize retention with premium offers, early access, and a "
            "personalized loyalty experience."
        ),
        "Budget-conscious customers": (
            "Test targeted bundles and controlled discounts while monitoring "
            "margin; avoid broad promotions."
        ),
        "Low-engagement customers": (
            "Use reactivation campaigns, reminders, and feedback prompts before "
            "considering win-back spend."
        ),
        "Loyal customers": (
            "Reward repeat purchases, add referral benefits, and use "
            "cross-sell offers matched to purchase history."
        ),
        "Established customers": (
            "Strengthen relevance with personalized content and periodic "
            "lifecycle offers."
        ),
        "Balanced customers": (
            "Use balanced, general campaigns and test channels before "
            "allocating larger budgets."
        ),
    }
    return recommendations[name]

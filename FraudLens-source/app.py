from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fraud_detector import (
    MODEL_NAMES,
    explain_transaction,
    make_demo_data,
    score_transactions,
    threshold_analysis,
    train_detector,
)

st.set_page_config(
    page_title="FraudLens · Transaction intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#e8edf5; --muted:#8c99ad; --mint:#75f0c0; --panel:#111a26; }
      html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
      .stApp { background: radial-gradient(ellipse at 72% 0%,#173044 0%,#0b111a 43%,#0b111a 100%); color:var(--ink); }
      [data-testid="stSidebar"] { background:#0e1620; border-right:1px solid #1d2a38; }
      [data-testid="stMetric"] { background:linear-gradient(145deg,#131e2b,#101823); border:1px solid #233343; padding:18px 20px; border-radius:14px; }
      [data-testid="stMetricLabel"] { color:#9eacbd; }
      [data-testid="stMetricValue"] { color:#eef5fc; font-weight:700; }
      .hero { padding:20px 0 12px; }
      .eyebrow { color:#75f0c0; font-family:'DM Mono',monospace; font-size:12px; letter-spacing:2px; text-transform:uppercase; }
      .hero h1 { font-size:clamp(34px,5vw,54px); line-height:1.07; letter-spacing:-2px; margin:10px 0 12px; color:#f2f6fb; }
      .hero p { color:#a8b5c5; max-width:760px; font-size:16px; line-height:1.7; }
      div.stButton > button[kind="primary"] { background:#75f0c0; color:#0b1714; border:0; font-weight:800; border-radius:10px; }
      .note { border:1px solid #704f25; background:#2a2115; color:#f2d9a9; border-radius:12px; padding:14px 16px; line-height:1.55; }
      .stTabs [data-baseweb="tab"] { color:#9eacbd; }
      .stTabs [aria-selected="true"] { color:#75f0c0 !important; }
      footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in {
    "training": None,
    "threshold": 0.5,
    "source_data": None,
    "data_origin": "uploaded dataset",
    "active_model": "Logistic Regression",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.sidebar.markdown("## ◈ FraudLens")
st.sidebar.caption("TRANSACTION INTELLIGENCE / PROTOTYPE")
st.sidebar.markdown("### 01 · Train and compare")
training_file = st.sidebar.file_uploader(
    "Labeled transaction CSV", type=["csv"], key="training_csv",
    help="Include numeric transaction features and a binary 0/1 fraud label.",
)
if training_file is not None:
    try:
        training_frame = pd.read_csv(training_file)
        targets = [column for column in training_frame.columns if column.lower() in {"class", "fraud", "is_fraud", "target", "label"}]
        numeric_columns = training_frame.select_dtypes(include="number").columns.tolist()
        target_options = list(dict.fromkeys(targets + numeric_columns))
        if target_options:
            guessed_target = targets[0] if targets else numeric_columns[-1]
            selected_target = st.sidebar.selectbox(
                "Label column", target_options, index=target_options.index(guessed_target)
            )
            if st.sidebar.button("Train and compare models", type="primary", use_container_width=True):
                with st.spinner("Training three classifiers and evaluating a shared holdout…"):
                    st.session_state.training = train_detector(training_frame, selected_target)
                    st.session_state.source_data = training_frame.copy()
                    st.session_state.data_origin = "uploaded dataset"
                    st.session_state.active_model = "Logistic Regression"
                st.sidebar.success("Three models trained and evaluated.")
        else:
            st.sidebar.error("No numeric columns found in this CSV.")
    except Exception as error:
        st.sidebar.error(f"Could not train from this CSV: {error}")

if st.sidebar.button("Load artificial demo", use_container_width=True):
    try:
        with st.spinner("Preparing artificial demo transactions…"):
            demo = make_demo_data()
            st.session_state.training = train_detector(demo, "Class")
            st.session_state.source_data = demo
            st.session_state.data_origin = "synthetic demo data"
            st.session_state.active_model = "Logistic Regression"
        st.sidebar.success("Demo ready. These metrics do not represent real-world performance.")
    except Exception as error:
        st.sidebar.error(str(error))

training = st.session_state.training
if training is not None:
    st.sidebar.markdown("### 02 · Review settings")
    st.session_state.active_model = st.sidebar.selectbox(
        "Active model", MODEL_NAMES,
        index=MODEL_NAMES.index(st.session_state.active_model),
    )
    st.session_state.threshold = st.sidebar.slider(
        "Decision threshold", min_value=0.01, max_value=0.99,
        value=float(st.session_state.threshold), step=0.01,
        help="Lower thresholds generally catch more fraud but raise more false alerts.",
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">◈ &nbsp; RISK SIGNALS, MADE VISIBLE</div>
      <h1>Every transaction<br>has a story.</h1>
      <p>Explore transaction patterns, compare baseline models, examine threshold tradeoffs, estimate review costs, and inspect local prediction signals.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="note"><strong>Decision-support prototype.</strong> Predictions and estimated costs are not verified financial outcomes. Keep a human reviewer in the loop. Use authorized, minimized, anonymized data only; never upload cardholder credentials.</div>',
    unsafe_allow_html=True,
)

if training is None:
    st.write("")
    st.markdown("### Start with a labeled CSV")
    st.markdown("Upload numeric transaction features and a binary label (`0` = legitimate, `1` = fraud), or load the clearly artificial demo from the sidebar. The app compares three models on the same stratified holdout.")
    st.code("Amount, Time, V1, V2, ..., Class\n149.62, 0, -1.36, -0.07, ..., 0\n2.10, 1, 1.19, 0.27, ..., 1", language="text")
    st.stop()

source_data = st.session_state.source_data
active_model_name = st.session_state.active_model
active_model = training.models[active_model_name]
probabilities = training.holdout_probabilities[active_model_name]
threshold = st.session_state.threshold

st.caption(f"{st.session_state.data_origin} · {training.metrics['transactions']:,} rows · {len(training.feature_columns)} numeric features · Active: {active_model_name}")
tabs = st.tabs(["Explore", "Model comparison", "Threshold & impact", "Explain", "Batch scan"])

with tabs[0]:
    st.markdown("### Explore the labeled transactions")
    fraud_rate = float(source_data[training.target_column].mean())
    fraud_rows = source_data[source_data[training.target_column] == 1]
    legit_rows = source_data[source_data[training.target_column] == 0]
    prevalence_cols = st.columns(3)
    prevalence_cols[0].metric("Fraud prevalence", f"{fraud_rate:.2%}")
    prevalence_cols[1].metric("Fraud transactions", f"{len(fraud_rows):,}")
    prevalence_cols[2].metric("Legitimate transactions", f"{len(legit_rows):,}")
    st.caption("Dataset-level descriptive statistics; demo results are artificial.")

    numeric_features = training.feature_columns
    amount_candidates = [column for column in numeric_features if column.lower() in {"amount", "transaction_amount", "value"}]
    amount_column = amount_candidates[0] if amount_candidates else st.selectbox("Feature to inspect", numeric_features, key="explore_feature")
    st.markdown(f"#### {amount_column}: distributions by label")
    distribution = source_data[[amount_column, training.target_column]].copy()
    distribution[amount_column] = pd.to_numeric(distribution[amount_column], errors="coerce")
    distribution = distribution.dropna(subset=[amount_column])
    if not distribution.empty:
        lower, upper = distribution[amount_column].quantile([0.01, 0.99])
        if lower == upper:
            lower, upper = distribution[amount_column].min(), distribution[amount_column].max() + 1
        distribution["Range"] = pd.cut(distribution[amount_column].clip(lower, upper), bins=25, duplicates="drop")
        hist = distribution.groupby(["Range", training.target_column], observed=False).size().unstack(fill_value=0)
        hist.columns = ["Legitimate" if label == 0 else "Fraud" for label in hist.columns]
        hist.index = hist.index.astype(str)
        st.bar_chart(hist, use_container_width=True)
        st.caption("Values are clipped at the 1st/99th percentiles for chart readability; counts outside that range remain represented in the edge bins.")

    st.markdown("#### Fraud vs. legitimate feature patterns")
    summary = source_data.groupby(training.target_column)[numeric_features].agg(["median", "mean"]).T
    summary.columns = ["Fraud" if label == 1 else "Legitimate" for label in summary.columns]
    st.dataframe(summary, use_container_width=True)
    explore_left, explore_right = st.columns(2)
    with explore_left:
        st.markdown("#### Feature correlations")
        corr = source_data[numeric_features + [training.target_column]].corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)
    with explore_right:
        st.markdown("#### IQR outlier counts")
        outlier_rows = []
        for feature in numeric_features:
            values = pd.to_numeric(source_data[feature], errors="coerce").dropna()
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()) if iqr else 0
            outlier_rows.append({"Feature": feature, "IQR outliers": count, "Share": count / max(len(values), 1)})
        st.dataframe(pd.DataFrame(outlier_rows).sort_values("IQR outliers", ascending=False), use_container_width=True, hide_index=True)
        st.caption("IQR is a simple screening heuristic; outliers may be valid transactions, not fraud.")

with tabs[1]:
    st.markdown("### Compare models on the same holdout")
    st.caption("All models use the same stratified 25% test split. Metrics at 0.50 are illustrative and may not fit operational costs.")
    comparison = pd.DataFrame(training.model_metrics).T
    comparison.index.name = "Model"
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
    st.markdown("#### Metric comparison")
    st.bar_chart(comparison[["pr_auc", "precision", "recall", "roc_auc"]], use_container_width=True)
    selected_metrics = training.model_metrics[active_model_name]
    mcols = st.columns(4)
    mcols[0].metric("PR-AUC", f"{selected_metrics['pr_auc']:.3f}")
    mcols[1].metric("Recall @ 0.50", f"{selected_metrics['recall']:.1%}")
    mcols[2].metric("Precision @ 0.50", f"{selected_metrics['precision']:.1%}")
    mcols[3].metric("ROC-AUC", f"{selected_metrics['roc_auc']:.3f}")
    st.info("Model choice is not automatic: compare precision/recall with the cost analysis and validate on time-separated data before relying on any model.")

with tabs[2]:
    st.markdown("### Threshold tradeoffs and estimated impact")
    st.caption("These counts describe only the held-out test rows. Cost projections depend entirely on the assumptions below.")
    amount_features = [column for column in training.feature_columns if column.lower() in {"amount", "transaction_amount", "value"}]
    observed_loss = float(pd.to_numeric(fraud_rows[amount_features[0]], errors="coerce").median()) if amount_features and len(fraud_rows) else 500.0
    if not np.isfinite(observed_loss) or observed_loss <= 0:
        observed_loss = 500.0
    cost_cols = st.columns(2)
    avg_loss = cost_cols[0].number_input("Average loss per missed fraud ($)", min_value=0.0, value=float(observed_loss), step=10.0, help="Default is the median labeled-fraud transaction amount, when available. Adjust to your validated loss estimate.")
    review_cost = cost_cols[1].number_input("Cost per false alert ($)", min_value=0.0, value=5.0, step=1.0, help="Estimated analyst/review cost for one legitimate transaction sent for review.")
    curve = threshold_analysis(training.holdout_labels, probabilities, average_fraud_loss=avg_loss, false_alert_cost=review_cost)
    current_idx = (curve["Threshold"] - threshold).abs().idxmin()
    current = curve.loc[current_idx]
    optimal = curve.loc[curve["Estimated operating cost"].idxmin()]

    st.markdown("#### Threshold → precision and recall")
    st.line_chart(curve.set_index("Threshold")[["Precision", "Recall"]], use_container_width=True)
    st.markdown("#### Threshold → false alerts and fraud caught")
    st.line_chart(curve.set_index("Threshold")[["False positives", "Fraud caught"]], use_container_width=True)
    impact_cols = st.columns(4)
    impact_cols[0].metric("Fraud caught", f"{int(current['Fraud caught']):,} / {int(training.holdout_labels.sum()):,}", help=f"{current['Recall']:.1%} of holdout fraud cases at this threshold.")
    impact_cols[1].metric("False alerts", f"{int(current['False positives']):,}")
    impact_cols[2].metric("Estimated missed-fraud loss", f"${current['Estimated loss']:,.0f}")
    impact_cols[3].metric("Cost-minimizing threshold", f"{optimal['Threshold']:.2f}", help="Minimum assumed missed-fraud loss plus false-alert handling cost across the plotted thresholds.")
    st.markdown(f"**At current threshold ({threshold:.2f}):** recall {current['Recall']:.1%}, precision {current['Precision']:.1%}, {int(current['False positives']):,} false alerts, estimated missed-fraud loss ${current['Estimated loss']:,.0f}.")
    st.markdown(f"**Estimated lowest-cost threshold:** {optimal['Threshold']:.2f} (estimated combined cost ${optimal['Estimated operating cost']:,.0f} on this holdout).")
    st.warning("Loss estimates are simplified: missed-fraud count × assumed average loss; false alerts × assumed review cost. They omit recoveries, overlap, customer impact, model drift, and many real costs. The holdout is not a forecast.")

with tabs[3]:
    st.markdown("### Explain a single transaction")
    st.caption("Factor signals are model-specific local explanations, not causal reasons or guarantees. Numeric inputs should use the training features' exact definitions and units.")
    with st.form("explain_transaction"):
        explain_cols = st.columns(2)
        transaction = {}
        for index, feature in enumerate(training.feature_columns):
            default = float(training.feature_defaults.get(feature, 0.0))
            transaction[feature] = explain_cols[index % 2].number_input(feature, value=default, format="%.6f", key=f"explain_{feature}")
        submitted = st.form_submit_button("Score and explain", type="primary")
    if submitted:
        row = pd.DataFrame([transaction], columns=training.feature_columns)
        probability = float(active_model.predict_proba(row)[0, 1])
        left, right = st.columns([1, 2])
        left.metric("Estimated fraud probability", f"{probability:.1%}")
        if probability >= threshold:
            right.error(f"Review recommended · score is above the {threshold:.0%} threshold.")
        else:
            right.success(f"Below threshold · score is below the {threshold:.0%} threshold.")
        st.progress(min(max(probability, 0.0), 1.0))
        explanation = explain_transaction(active_model, row, training.holdout_features, training.feature_columns)
        explanation["Direction"] = np.where(explanation["Influence"] >= 0, "Raises score", "Lowers score")
        st.markdown("#### Largest model signals")
        st.dataframe(explanation[["Feature", "Value", "Direction", "Influence"]].head(12), use_container_width=True, hide_index=True)
        st.bar_chart(explanation.head(12).set_index("Feature")[["Influence"]], use_container_width=True)
        if active_model_name == "Logistic Regression":
            st.caption("Logistic influences are signed contributions in standardized log-odds space relative to the model intercept.")
        else:
            st.caption("Tree-model influences are one-feature-at-a-time probability changes from the median holdout transaction; interactions mean these effects are not additive.")
        st.caption("Model scores may not be calibrated probabilities. Do not make an adverse financial decision from this output alone.")

with tabs[4]:
    st.markdown("### Batch scan")
    st.caption(f"Upload transactions with the same numeric features. Active model: {active_model_name}.")
    batch_file = st.file_uploader("Transaction batch CSV", type=["csv"], key="batch_csv")
    if batch_file is not None:
        try:
            batch = pd.read_csv(batch_file)
            scored = score_transactions(active_model, batch, training.feature_columns, threshold)
            flagged_count = int(scored["Fraud prediction"].sum())
            first, second, third = st.columns(3)
            first.metric("Transactions scanned", f"{len(scored):,}")
            second.metric("Review recommended", f"{flagged_count:,}")
            third.metric("Threshold", f"{threshold:.0%}")
            st.dataframe(scored.sort_values("Fraud probability", ascending=False), use_container_width=True, hide_index=True)
            st.download_button(
                "Download scored CSV", data=scored.to_csv(index=False).encode("utf-8"),
                file_name="fraud_scan_results.csv", mime="text/csv", type="primary",
            )
        except Exception as error:
            st.error(f"Could not score this file: {error}")

st.divider()
st.caption("FraudLens is an educational prototype. Keep humans in the loop; validate independently; do not use synthetic demo metrics as evidence.")

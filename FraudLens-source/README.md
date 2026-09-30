# FraudLens — Credit Card Fraud Detector

FraudLens is an educational Streamlit prototype for exploring transaction-fraud classification. It compares logistic regression, random forest, and gradient boosting on one shared holdout; explores labeled transaction patterns; analyzes threshold tradeoffs and configurable cost assumptions; explains local transaction scores; and scores batch files.

> **Not for production decisions.** This prototype is not a payment control or financial advice. Model outputs are uncertain signals and require human review. Do not upload raw cardholder data or any data you are not authorized to process. Use minimized, anonymized data in a properly secured environment.

## Run locally

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. In the sidebar, load the artificial demo or upload your own labeled CSV.

## Data format

The training file must contain numeric transaction feature columns and a binary target with both classes represented:

- `0` means legitimate transaction.
- `1` means fraud.
- Common target names are detected automatically: `Class`, `Fraud`, `Is_Fraud`, `target`, and `label`. You can select a different numeric target in the sidebar.
- At least 8 rows and 2 examples of each class are required. Missing numeric feature values are imputed with training medians.
- Examples of features include `Amount`, `Time`, and anonymized variables such as `V1`–`V28`. Include the same feature columns (and definitions/units) when analyzing new transactions.

The batch scanner preserves source columns and adds `Fraud probability` and `Fraud prediction`. Its prediction column uses the threshold selected in the sidebar.

## Evaluation notes

The app reserves one stratified 25% holdout shared across the three models. It reports precision, recall, PR-AUC, and ROC-AUC, primarily emphasizing PR-AUC for imbalanced fraud labels. The threshold view charts precision, recall, false positives, and fraud caught across cutoffs. Financial estimates use user-editable assumptions for average loss per missed fraud and false-alert review cost. The displayed cost-minimizing threshold minimizes only those assumed costs on the holdout; it is not a production recommendation.

Exploration includes fraud prevalence, feature distributions, fraud-vs-legitimate mean/median patterns, correlations, and IQR-based outlier counts. Explanations use standardized logistic-regression contributions or tree-model one-feature-at-a-time probability changes. These are model signals, not causal explanations; tree effects are not additive. A stratified random split can overstate performance when transactions are related or time-dependent. For a real evaluation, use a leakage-resistant, time-aware split and independently validate calibration, drift, operational costs, fairness, and alert capacity.

The **Load artificial demo** button creates toy data with intentionally generated patterns so you can explore the interface. Its metrics do not represent real transaction data and must not be used to assess fraud performance.

## Tests

Run `python -m unittest discover -s tests`.

## Limitations

- The model is a baseline, not a bank-grade fraud platform.
- Scores are not guaranteed to be calibrated probabilities.
- Feature quality, label quality, sampling, class prevalence, and distribution shift can materially change results.
- No real card-number data is needed or accepted as a useful feature; avoid handling sensitive payment credentials.
- Do not automate declines, account actions, or investigations using this demo.

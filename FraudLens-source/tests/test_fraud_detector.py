import unittest

import pandas as pd

from fraud_detector import (
    MODEL_NAMES,
    explain_transaction,
    make_demo_data,
    score_transactions,
    threshold_analysis,
    train_detector,
    validate_training_data,
)


class FraudDetectorTests(unittest.TestCase):
    def test_demo_data_can_train_and_score(self):
        data = make_demo_data(rows=500, random_state=7)
        result = train_detector(data, "Class")
        self.assertIn("recall", result.metrics)
        self.assertIn("pr_auc", result.metrics)
        self.assertEqual(set(result.models), set(MODEL_NAMES))
        self.assertEqual(set(result.model_metrics), set(MODEL_NAMES))

        scored = score_transactions(
            result.model,
            data.drop(columns="Class").head(5),
            result.feature_columns,
        )
        self.assertEqual(len(scored), 5)
        self.assertIn("Fraud probability", scored.columns)
        self.assertTrue(scored["Fraud probability"].between(0, 1).all())

    def test_threshold_analysis_calculates_counts_and_cost(self):
        result = threshold_analysis(
            labels=[1, 1, 0, 0],
            probabilities=[0.9, 0.3, 0.8, 0.1],
            thresholds=[0.5],
            average_fraud_loss=100,
            false_alert_cost=10,
        ).iloc[0]
        self.assertEqual(result["Fraud caught"], 1)
        self.assertEqual(result["False positives"], 1)
        self.assertEqual(result["Fraud missed"], 1)
        self.assertEqual(result["Estimated loss"], 100)
        self.assertEqual(result["Estimated operating cost"], 110)

    def test_local_explanation_returns_each_feature(self):
        data = make_demo_data(rows=300, random_state=11)
        result = train_detector(data)
        row = data[result.feature_columns].iloc[[0]]
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                explanation = explain_transaction(
                    result.models[model_name],
                    row,
                    result.holdout_features,
                    result.feature_columns,
                )
                self.assertEqual(set(explanation["Feature"]), set(result.feature_columns))
                self.assertTrue(explanation["Influence"].notna().all())

    def test_target_detection_is_case_insensitive(self):
        data = make_demo_data(rows=200)
        data = data.rename(columns={"Class": "class"})
        target, features = validate_training_data(data)
        self.assertEqual(target, "class")
        self.assertNotIn(target, features)

    def test_training_requires_both_classes(self):
        data = make_demo_data(rows=200)
        data["Class"] = 0
        with self.assertRaisesRegex(ValueError, "both 0 .* and 1"):
            train_detector(data)

    def test_batch_requires_all_trained_features(self):
        data = make_demo_data(rows=300)
        result = train_detector(data)
        with self.assertRaisesRegex(ValueError, "Missing required feature columns"):
            score_transactions(result.model, pd.DataFrame({"Amount": [10.0]}), result.feature_columns)


if __name__ == "__main__":
    unittest.main()

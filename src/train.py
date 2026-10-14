"""Train early-detection classifiers on featurized job postings.

We train two Spark MLlib classifiers on the same feature vector:

  - LogisticRegression: a solid baseline, easy to interpret. The model
    coefficients tell us which terms push a posting toward "emerging".
  - RandomForest: picks up interactions between features that a linear
    model would miss.

We evaluate both on a holdout split (accuracy, F1, AUC) and save the
better one by F1 to the model output directory.

Run:
    python src/train.py --input data/sample_jobs.json --model-output models/
"""

import argparse
from pathlib import Path

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import functions as F

from config import SETTINGS, spark_session
from preprocess import add_text_features, build_pipeline


def split_data(df, label_col: str):
    """Split into train/test, keeping the split deterministic.

    On tiny demo data the split can be lopsided — that is fine here,
    since the goal is to exercise the full pipeline.
    """
    train, test = df.randomSplit(
        [SETTINGS.train_ratio, 1 - SETTINGS.train_ratio],
        seed=SETTINGS.random_seed,
    )
    # Make sure both classes appear in training when possible.
    return train, test


def evaluate(model, test_df, label_col: str) -> dict:
    """Compute accuracy, F1, and AUC on the test set."""
    preds = model.transform(test_df)

    acc = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy"
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    ).evaluate(preds)
    auc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(preds)
    return {"accuracy": acc, "f1": f1, "auc": auc}


def train_label(df, label_col: str):
    """Featurize and train both classifiers for one label column."""
    df = add_text_features(df)
    pipeline = build_pipeline()
    featurizer = pipeline.fit(df)
    featured = featurizer.transform(df)

    train, test = split_data(featured, label_col)
    print(f"Label '{label_col}': train={train.count()}, test={test.count()}")

    # Logistic regression baseline.
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=SETTINGS.logistic_max_iter,
        regParam=SETTINGS.logistic_reg_param,
    )
    lr_model = lr.fit(train)
    lr_metrics = evaluate(lr_model, test, label_col)

    # Random forest for non-linear patterns.
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=label_col,
        numTrees=SETTINGS.forest_num_trees,
        maxDepth=SETTINGS.forest_max_depth,
        seed=SETTINGS.random_seed,
    )
    rf_model = rf.fit(train)
    rf_metrics = evaluate(rf_model, test, label_col)

    print(f"  LogisticRegression: {lr_metrics}")
    print(f"  RandomForest:       {rf_metrics}")

    # Keep the better model by F1.
    if rf_metrics["f1"] >= lr_metrics["f1"]:
        best, best_name, best_metrics = rf_model, "random_forest", rf_metrics
    else:
        best, best_name, best_metrics = lr_model, "logistic_regression", lr_metrics

    print(f"  -> best: {best_name} (f1={best_metrics['f1']:.3f})")
    return featurizer, best, best_name, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train job detection models")
    parser.add_argument("--input", default=SETTINGS.sample_data_path)
    parser.add_argument("--model-output", default=SETTINGS.model_output)
    args = parser.parse_args()

    out_dir = Path(args.model_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = spark_session("job-train")
    try:
        df = spark.read.json(args.input)
        df = df.withColumn("is_emerging", F.col("is_emerging").cast("double"))
        df = df.withColumn("is_urgent", F.col("is_urgent").cast("double"))
        print(f"Loaded {df.count()} postings from {args.input}")

        results = {}
        for label in ("is_emerging", "is_urgent"):
            featurizer, best, name, metrics = train_label(df, label)
            label_dir = out_dir / label
            featurizer.write().overwrite().save(str(label_dir / "featurizer"))
            best.write().overwrite().save(str(label_dir / "model"))
            results[label] = {"model": name, **metrics}

        print("Saved models to", out_dir)
        print("Summary:", results)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

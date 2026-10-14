"""Score new job postings in real time with Spark Structured Streaming.

How it works:
  - Watches an input directory for new JSON files (file source — easy to
    demo locally). In production you would swap this for Kafka.
  - Applies the same preprocessing + TF-IDF featurizer used in training.
  - Loads the saved classifiers and scores each posting for
    "is_emerging" and "is_urgent".
  - Writes scored rows to the console (and optionally to Parquet).

Run:
    python src/streaming_detect.py --input data/stream_in --model models/

Then drop JSON files with the same schema as data/sample_jobs.json into
data/stream_in/ and watch the scores appear.
"""

import argparse
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.classification import ClassificationModel
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from config import SETTINGS, spark_session
from preprocess import build_pipeline


# Explicit schema keeps streaming robust — a malformed file fails fast
# with a clear error instead of silently producing null columns.
JOB_SCHEMA = StructType(
    [
        StructField("id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("company", StringType(), True),
        StructField("location", StringType(), True),
        StructField("category", StringType(), True),
        StructField("description", StringType(), True),
        StructField("skills", ArrayType(StringType()), True),
        StructField("posted_date", StringType(), True),
        StructField("is_remote", StringType(), True),
    ]
)


def score_stream(spark, input_dir: str, model_dir: str):
    """Build the streaming DataFrame with both model scores attached."""
    raw = (
        spark.readStream.schema(JOB_SCHEMA)
        .option("maxFilesPerTrigger", SETTINGS.stream_max_files_per_trigger)
        .json(input_dir)
    )

    # Reuse the training-time preprocessing so features match exactly.
    # The featurizer was fit on training data and saved by train.py.
    featurizer = PipelineModel.load(str(Path(model_dir) / "is_emerging" / "featurizer"))
    featured = featurizer.transform(raw)

    emerging_model = ClassificationModel.load(
        str(Path(model_dir) / "is_emerging" / "model")
    )
    urgent_model = ClassificationModel.load(
        str(Path(model_dir) / "is_urgent" / "model")
    )

    scored = emerging_model.transform(featured)
    scored = scored.withColumnRenamed("prediction", "emerging_pred")
    scored = scored.withColumnRenamed("probability", "emerging_prob")

    scored = urgent_model.transform(scored)
    scored = scored.withColumnRenamed("prediction", "urgent_pred")

    # Pull out the positive-class probability for readability.
    get_prob = F.udf(lambda v: float(v[1]) if v is not None else 0.0, DoubleType())
    scored = scored.withColumn("emerging_score", get_prob(F.col("emerging_prob")))

    return scored.select(
        "id",
        "title",
        "company",
        "emerging_pred",
        "emerging_score",
        "urgent_pred",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-score job postings")
    parser.add_argument("--input", default=SETTINGS.stream_input_dir)
    parser.add_argument("--model", default=SETTINGS.model_output)
    parser.add_argument("--output", default=SETTINGS.stream_output_dir)
    args = parser.parse_args()

    Path(args.input).mkdir(parents=True, exist_ok=True)
    Path(SETTINGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    spark = spark_session("job-streaming-detect")
    spark.sparkContext.setLogLevel("WARN")
    try:
        scored = score_stream(spark, args.input, args.model)

        console_query = (
            scored.writeStream.outputMode("append")
            .format("console")
            .option("truncate", "60")
            .trigger(processingTime=f"{SETTINGS.stream_trigger_seconds} seconds")
            .start()
        )

        parquet_query = (
            scored.writeStream.outputMode("append")
            .format("parquet")
            .option("path", args.output)
            .option("checkpointLocation", SETTINGS.checkpoint_dir)
            .trigger(processingTime=f"{SETTINGS.stream_trigger_seconds} seconds")
            .start()
        )

        print(f"Watching {args.input} — drop JSON files there to score them.")
        console_query.awaitTermination()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""Shared configuration for the early job detection pipeline.

Keeping settings in one place makes it easy to run the same code locally
and on a cluster — you only change paths and Spark options here, not in
every script.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """All tunable settings for the pipeline."""

    # Where to read / write data. Override with env vars in production.
    sample_data_path: str = "data/sample_jobs.json"
    fetched_data_path: str = "data/fetched_jobs.json"
    features_output: str = "data/features"
    model_output: str = "models"
    stream_input_dir: str = "data/stream_in"
    stream_output_dir: str = "data/stream_out"
    checkpoint_dir: str = "checkpoints/job_stream"

    # Placeholder API — replace with a real, permitted source.
    # We deliberately default to an empty URL so the scraper falls back
    # to sample data instead of hitting a real board by accident.
    jobs_api_url: str = os.getenv("JOBS_API_URL", "")
    jobs_api_key: str = os.getenv("JOBS_API_KEY", "")
    request_timeout_seconds: int = 15
    max_requests_per_minute: int = 20

    # Spark
    app_name: str = "early-job-detection"
    spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
    spark_shuffle_partitions: int = 8

    # Text features
    tfidf_vocab_size: int = 5000
    tfidf_min_df: int = 1
    max_title_length: int = 200
    max_description_length: int = 10000

    # Model training
    train_ratio: float = 0.8
    random_seed: int = 42
    logistic_max_iter: int = 50
    logistic_reg_param: float = 0.1
    forest_num_trees: int = 100
    forest_max_depth: int = 8

    # Streaming
    stream_trigger_seconds: int = 10
    stream_max_files_per_trigger: int = 5

    # Urgency keywords used only for the demo labels.
    # A real system would learn this from time-to-fill outcomes.
    urgency_keywords: list = field(
        default_factory=lambda: [
            "urgent",
            "immediate",
            "asap",
            "immediately",
            "quickly",
            "fast-paced hiring",
        ]
    )

    # Skills we track for the trends notebook.
    tracked_skills: list = field(
        default_factory=lambda: [
            "python",
            "spark",
            "kafka",
            "sql",
            "airflow",
            "dbt",
            "aws",
            "docker",
            "kubernetes",
            "machine learning",
            "llm",
            "genai",
        ]
    )


# A single shared instance — import this in the other modules.
SETTINGS = Settings()


def spark_session(app_name: str | None = None):
    """Build a local Spark session with sensible defaults.

    We keep this helper here so every script gets the same config
    without copy-pasting builder code.
    """
    from pyspark.sql import SparkSession

    name = app_name or SETTINGS.app_name
    return (
        SparkSession.builder.appName(name)
        .master(SETTINGS.spark_master)
        .config(
            "spark.sql.shuffle.partitions",
            str(SETTINGS.spark_shuffle_partitions),
        )
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

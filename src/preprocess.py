"""Preprocess job postings with Spark: clean text and build TF-IDF features.

Steps:
  1. Load JSON postings into a Spark DataFrame
  2. Clean title + description (lowercase, strip HTML, collapse whitespace)
  3. Tokenize, remove stopwords, build TF-IDF vectors
  4. Add numeric features (description length, skill count)
  5. Assemble everything into a single "features" vector for ML

Run:
    python src/preprocess.py --input data/sample_jobs.json --output data/features
"""

import argparse
import re

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    RegexTokenizer,
    StopWordsRemover,
    VectorAssembler,
)
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from config import SETTINGS, spark_session


HTML_TAG = re.compile(r"<[^>]+>")
MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Lowercase, strip HTML, and collapse whitespace.

    We keep it simple here — heavy NLP (lemmatization, etc.) lives
    in the notebook if you want to experiment with it.
    """
    if not text:
        return ""
    text = HTML_TAG.sub(" ", text)
    text = text.lower()
    text = MULTI_SPACE.sub(" ", text)
    return text.strip()


def add_text_features(df):
    """Add cleaned text, token, and numeric feature columns."""
    clean_udf = F.udf(clean_text, StringType())

    # Combine title + description so the model sees both.
    df = df.withColumn(
        "combined_text",
        F.concat_ws(" ", F.col("title"), F.col("description")),
    )
    df = df.withColumn("cleaned_text", clean_udf(F.col("combined_text")))

    # Numeric signals that are cheap and often useful.
    df = df.withColumn("desc_length", F.length(F.col("description")))
    df = df.withColumn("skill_count", F.size(F.col("skills")))

    # A simple array of lowercase skills, handy for the trends notebook.
    lower_skills = F.udf(
        lambda skills: [str(s).lower() for s in (skills or [])],
        ArrayType(StringType()),
    )
    df = df.withColumn("skills_lower", lower_skills(F.col("skills")))
    return df


def build_pipeline() -> Pipeline:
    """Build the Spark ML pipeline for tokenization + TF-IDF."""
    tokenizer = RegexTokenizer(
        inputCol="cleaned_text",
        outputCol="tokens",
        pattern="\\W+",
        minTokenLength=2,
    )
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    hashing = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        numFeatures=SETTINGS.tfidf_vocab_size,
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

    # Combine TF-IDF with numeric features into one vector.
    assembler = VectorAssembler(
        inputCols=["tfidf_features", "desc_length", "skill_count"],
        outputCol="features",
    )
    return Pipeline(stages=[tokenizer, remover, hashing, idf, assembler])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess job postings")
    parser.add_argument("--input", default=SETTINGS.sample_data_path)
    parser.add_argument("--output", default=SETTINGS.features_output)
    args = parser.parse_args()

    spark = spark_session("job-preprocess")
    try:
        df = spark.read.json(args.input)
        print(f"Loaded {df.count()} postings from {args.input}")

        df = add_text_features(df)
        pipeline = build_pipeline()
        model = pipeline.fit(df)
        featured = model.transform(df)

        # Keep a tidy subset for inspection and for train.py.
        out = featured.select(
            "id",
            "title",
            "company",
            "category",
            "is_emerging",
            "is_urgent",
            "features",
            "skills_lower",
        )
        out.write.mode("overwrite").parquet(args.output)
        print(f"Wrote featurized data to {args.output}")
        out.select("title", "is_emerging", "is_urgent").show(truncate=60)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

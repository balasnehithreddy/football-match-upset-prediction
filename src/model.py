"""
model.py
Train a Logistic Regression (PySpark ML) to predict home_win.
Write Gold outputs: predictions, metrics.json, and top_upsets (p>=0.70 but home did not win).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array  # <-- key fix

SILVER_PATH = Path("data/silver/all_features.parquet")
GOLD_DIR = Path("data/gold")
UPSET_THRESHOLD = 0.70

FEATURES = [
    "home_points_last5",
    "away_points_last5",
    "home_gf_last5",
    "home_ga_last5",
    "away_gf_last5",
    "away_ga_last5",
]

def spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("football-model")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

def time_split(df: DataFrame, ratio: float = 0.7) -> Tuple[DataFrame, DataFrame]:
    """Strict time-based split using row_number over date."""
    w = Window.orderBy("date")
    numbered = df.withColumn("rn", F.row_number().over(w))
    n = numbered.count()
    cut = int(n * ratio)
    train = numbered.where(F.col("rn") <= cut).drop("rn")
    test  = numbered.where(F.col("rn") >  cut).drop("rn")
    return train, test

def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    spark_session = spark()

    if not SILVER_PATH.exists():
        raise FileNotFoundError("Silver data not found. Run transform.py first.")

    base = (
        spark_session.read.parquet(str(SILVER_PATH))
        .withColumn("date", F.to_date("date"))
        .orderBy("date")
    )

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    data = assembler.transform(base).select(
        "competition", "season", "date", "match_id",
        "home_team", "away_team", "home_goals", "away_goals",
        "home_win", "features",
    )

    train, test = time_split(data, ratio=0.7)

    lr = LogisticRegression(featuresCol="features", labelCol="home_win", maxIter=100)
    model = lr.fit(train)

    pred = model.transform(test)
    # Convert vector -> array, then take index 1 (P(home_win=1))
    pred = pred.withColumn("prob_arr", vector_to_array("probability"))
    pred = pred.withColumn("p_home_win", F.col("prob_arr")[1]).drop("prob_arr")

    # metrics
    evaluator = BinaryClassificationEvaluator(labelCol="home_win", rawPredictionCol="rawPrediction")
    auc = float(evaluator.evaluate(pred))
    acc = float(
        pred.select((F.col("prediction") == F.col("home_win")).cast("int").alias("ok"))
            .agg(F.avg("ok"))
            .first()[0]
    )

    # predictions → Gold
    predictions = (
        pred.select(
            "competition", "season", "date", "match_id",
            "home_team", "away_team", "home_goals", "away_goals",
            F.col("home_win").alias("actual_result"),
            F.col("prediction").cast("int").alias("predicted_result"),
            F.round("p_home_win", 4).alias("p_home_win"),
        )
        .orderBy("date")
    )
    (predictions.coalesce(1)
               .write.mode("overwrite")
               .parquet(str(GOLD_DIR / "predictions.parquet")))
    (predictions.coalesce(1)
               .write.mode("overwrite")
               .option("header", True)
               .csv(str(GOLD_DIR / "predictions_csv")))

    # metrics.json
    write_json(GOLD_DIR / "metrics.json", {"accuracy": round(acc, 4), "auc": round(auc, 4)})

    # top upsets
    top_upsets = predictions.where(
        (F.col("p_home_win") >= UPSET_THRESHOLD) & (F.col("actual_result") == 0)
    ).orderBy(F.desc("p_home_win"))
    (top_upsets.coalesce(1)
               .write.mode("overwrite")
               .option("header", True)
               .csv(str(GOLD_DIR / "top_upsets_csv")))

    print(f"✓ Gold written. ACC={acc:.3f}  AUC={auc:.3f}")
    spark_session.stop()


if __name__ == "__main__":
    main()

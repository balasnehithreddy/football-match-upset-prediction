"""
transform.py
Transform Bronze match JSON into Silver features using PySpark.

Silver columns (minimal, human-readable):
- competition, season, date, match_id
- home_team, away_team
- home_goals, away_goals
- home_win (label: 1 if home wins else 0)
- home_points_last5, away_points_last5
- home_gf_last5, home_ga_last5
- away_gf_last5, away_ga_last5
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")


def spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("football-transform")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def read_bronze_file(spark: SparkSession, path: Path) -> DataFrame:
    """
    Read one *_raw.json file and select only the fields we need.
    Skips matches without final scores.
    """
    with path.open("r") as f:
        payload: Dict = json.load(f)

    rows: List[Dict] = []
    comp = payload.get("competition", {}).get("code")  # may be None in some payloads
    for m in payload.get("matches", []):
        score = (m.get("score") or {}).get("fullTime") or {}
        hg, ag = score.get("home"), score.get("away")
        if hg is None or ag is None:
            continue
        rows.append({
            "competition": comp or (m.get("competition") or {}).get("code", "UNK"),
            "season": (m.get("season") or {}).get("startDate", "")[:4],
            "match_id": m.get("id"),
            "date": (m.get("utcDate") or "")[:10],
            "home_team": (m.get("homeTeam") or {}).get("name"),
            "away_team": (m.get("awayTeam") or {}).get("name"),
            "home_goals": int(hg),
            "away_goals": int(ag),
        })

    df = spark.createDataFrame(rows)
    return df.withColumn("home_win", (F.col("home_goals") > F.col("away_goals")).cast("int"))


def build_recent_form(df: DataFrame) -> DataFrame:
    """
    Compute rolling last-5 stats per team, then join back to matches as home/away features.
    """
    # Long format (one row per team per match)
    home = (
        df.select(
            "competition", "season", "date", "match_id",
            F.col("home_team").alias("team"),
            F.lit("home").alias("venue"),
            F.col("home_goals").alias("gf"),
            F.col("away_goals").alias("ga"),
        )
        .withColumn(
            "points",
            F.when(F.col("gf") > F.col("ga"), 3)
             .when(F.col("gf") == F.col("ga"), 1)
             .otherwise(0),
        )
    )

    away = (
        df.select(
            "competition", "season", "date", "match_id",
            F.col("away_team").alias("team"),
            F.lit("away").alias("venue"),
            F.col("away_goals").alias("gf"),
            F.col("home_goals").alias("ga"),
        )
        .withColumn(
            "points",
            F.when(F.col("gf") > F.col("ga"), 3)
             .when(F.col("gf") == F.col("ga"), 1)
             .otherwise(0),
        )
    )

    long_df = home.unionByName(away)
    w = Window.partitionBy("team").orderBy("date").rowsBetween(-5, -1)

    stats = (
        long_df
        .withColumn("points_last5", F.sum("points").over(w))
        .withColumn("gf_last5", F.sum("gf").over(w))
        .withColumn("ga_last5", F.sum("ga").over(w))
        .where(F.col("points_last5").isNotNull())  # requires at least one prior match
    )

    # Join back to match rows (home/away splits)
    home_stats = (
        stats.filter(F.col("venue") == "home")
        .select(
            "match_id", "date",
            F.col("points_last5").alias("home_points_last5"),
            F.col("gf_last5").alias("home_gf_last5"),
            F.col("ga_last5").alias("home_ga_last5"),
        )
    )

    away_stats = (
        stats.filter(F.col("venue") == "away")
        .select(
            "match_id",
            F.col("points_last5").alias("away_points_last5"),
            F.col("gf_last5").alias("away_gf_last5"),
            F.col("ga_last5").alias("away_ga_last5"),
        )
    )

    return (
        df.join(home_stats, ["match_id", "date"], "inner")
          .join(away_stats, "match_id", "inner")
          .select(
              "competition", "season", "date", "match_id",
              "home_team", "away_team",
              "home_goals", "away_goals", "home_win",
              "home_points_last5", "away_points_last5",
              "home_gf_last5", "home_ga_last5",
              "away_gf_last5", "away_ga_last5",
          )
    )


def main() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    spark_session = spark()

    bronze_files = sorted(p for p in BRONZE_DIR.glob("*_raw.json"))
    if not bronze_files:
        print("No Bronze files found. Run extract.py first.")
        spark_session.stop()
        return

    dfs: List[DataFrame] = [read_bronze_file(spark_session, p) for p in bronze_files]
    base = dfs[0]
    for d in dfs[1:]:
        base = base.unionByName(d)

    features = build_recent_form(base).orderBy("date")

    # Write once (parquet for Spark; CSV for eyeballing)
    (features.coalesce(1)
             .write.mode("overwrite")
             .parquet(str(SILVER_DIR / "all_features.parquet")))
    (features.coalesce(1)
             .write.mode("overwrite")
             .option("header", True)
             .csv(str(SILVER_DIR / "all_features_csv")))

    print(f"✓ Silver written: {features.count()} rows.")
    features.select(
        "date", "home_team", "away_team", "home_points_last5", "away_points_last5"
    ).show(10, truncate=False)

    spark_session.stop()


if __name__ == "__main__":
    main()

# Lab 10: Distributed Machine Learning on the Cloud with Databricks

COM6012 Scalable Machine Learning 2026 ([github/COM6012/ScalableML](https://github.com/COM6012/ScalableML)) by Tahsin Khan at The University of Sheffield

## Study schedule

- [Section 1](#1-environment-setup-and-data-loading): To finish in the lab session. **Critical**
- [Section 2](#2-mlflow-experiment-tracking): To finish in the lab session. **Critical**
- [Section 3](#3-model-registration-and-batch-inference): To finish in the lab session. **Essential**
- [Section 4](#4-drift-detection-statistics-visualisation-and-interpretable-metrics): To finish in the lab session. **Essential**
- [Section 5](#5-incremental-inference-simulation-and-lakehouse-monitoring): To finish before the following session. ***Exercise***
- [Section 6](#6-retraining-on-combined-data): To explore further. *Optional*

## Introduction

In this lab we build a complete end-to-end machine learning pipeline on Databricks — from data loading and model training, through experiment tracking with MLflow and model registration in Unity Catalog, to drift detection and production monitoring.

**What are we predicting?** We predict whether a NYC Yellow Taxi trip will receive a generous tip, defined as a tip exceeding 20% of the fare amount. This is a **binary classification** task. The model is trained on January 2019 data (winter) and evaluated against June 2019 data (summer) — a realistic drift scenario in which seasonal changes in passenger behaviour, trip patterns, and payment habits all shift the underlying distribution.

| Element | Detail |
|---|---|
| Task type | Binary classification |
| Target | `tip_amount / fare_amount > 0.20` → label = 1 (generous) |
| Features | Trip distance, fare, passenger count, pickup hour, day of week, pickup zone, payment type, trip duration |
| Training period | January 2019 (winter baseline) |
| Production period | June 2019 (summer — drift scenario) |
| Metrics | AUC, Accuracy, F1, % predictions within acceptable range |

**Note on payment type.** Cash trips (`payment_type=2`) have `tip_amount=0` by convention — cash tips are not digitally recorded. This makes `payment_type` a very strong predictor and is worth bearing in mind when interpreting model outputs.

**Prerequisites.** You should have completed Labs 1–9. You will need a Databricks Free Edition account — sign up at [signup.databricks.com](https://signup.databricks.com) (no credit card required).

> **Databricks Free Edition (2026)**
> Databricks Community Edition was **retired on January 1, 2026** and replaced by **Free Edition**. Free Edition includes MLflow, Unity Catalog, Delta Lake, Python notebooks, and serverless compute. It uses **serverless compute only** — no cluster to configure or start. R and Scala are not supported. Daily compute quotas apply.
>
> **Important notes for Free Edition serverless:**
> - `spark` is **pre-provided** in every Databricks notebook — never call `SparkSession.builder.getOrCreate()`
> - `mlflow.spark.autolog()` is **not supported** on serverless clusters — parameters and metrics are logged manually in this lab
> - Serverless sessions can **time out** during long operations — if this happens, detach and reattach Serverless compute, then use Run All to re-execute the notebook from the top
> - All variables (including `df_train_raw`) are lost when a session resets completely — there is no way to recover without re-running from the first cell

## 1. Environment Setup and Data Loading

#### Step 1 — Create a notebook

1. Log in to your Databricks Free Edition workspace at [signup.databricks.com](https://signup.databricks.com)
2. In the left sidebar, click **+ New → Notebook**
3. Name it `lab10_[your_username]`, language: **Python**
4. In the **Connect** dropdown, select **Serverless** — this is the only compute option in Free Edition; it starts automatically with no configuration required

#### Step 2 — Load the NYC Yellow Taxi Dataset

The NYC Taxi and Limousine Commission (TLC) publishes all trip records at [nyc.gov/site/tlc](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). We load one month per period using **pandas**, which handles HTTPS URLs natively. Spark cannot read directly from HTTPS URLs because its HTTP filesystem does not implement the directory-listing operation that `spark.read` requires. We therefore download via pandas and convert to a Spark DataFrame.

Note that `spark` is already available in Databricks notebooks — we use it directly without any import or instantiation.

```python
import pandas as pd
from pyspark.sql.functions import col, hour, dayofweek, month, when, unix_timestamp

# spark is pre-provided in Databricks notebooks — do not call SparkSession.builder

base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}.parquet"

# Each file is ~150 MB — allow 30–60 seconds per download
print("Downloading Jan 2019 (training period)...")
df_train_raw = spark.createDataFrame(
    pd.read_parquet(base_url.format("2019-01"))
)

print("Downloading Jun 2019 (production/drift period)...")
df_production_raw = spark.createDataFrame(
    pd.read_parquet(base_url.format("2019-06"))
)

print(f"Training rows (Jan 2019):   {df_train_raw.count():,}")
print(f"Production rows (Jun 2019): {df_production_raw.count():,}")
df_train_raw.printSchema()
```

> **If internet access is unavailable in your workspace**, download the two files directly from your browser and upload them to a Unity Catalog Volume:
> 1. Download from your browser:
>    - `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet`
>    - `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-06.parquet`
> 2. In Databricks: **Catalog → your schema → Create Volume** → Upload both files
> 3. Then load with:
> ```python
> df_train_raw      = spark.read.parquet("/Volumes/ml/com6012/lab10_data/yellow_tripdata_2019-01.parquet")
> df_production_raw = spark.read.parquet("/Volumes/ml/com6012/lab10_data/yellow_tripdata_2019-06.parquet")
> ```

#### Step 3 — Feature Engineering and Sampling

> **Note:** The cells in this step will take **2–4 minutes** to run on Databricks Free Edition. This is expected behaviour. Spark uses lazy evaluation — no computation happens when transformations are defined. Execution is triggered by the first `.count()` call, at which point Spark applies all filters, computes new columns, splits the data, and samples it in a single pass. Allow the cell to complete before moving on.

```python
feature_cols = [
    "trip_distance", "fare_amount", "passenger_count",
    "pickup_hour", "pickup_dow", "pickup_month",
    "pickup_zone", "payment_type", "trip_duration_min"
]

def prepare_features(df):
    return (
        df
        .filter(col("fare_amount") > 2.5)
        .filter(col("trip_distance") > 0.1)
        .filter(col("passenger_count") > 0)
        .filter(col("tip_amount") >= 0)
        .withColumn("pickup_hour",       hour("tpep_pickup_datetime"))
        .withColumn("pickup_dow",        dayofweek("tpep_pickup_datetime"))
        .withColumn("pickup_month",      month("tpep_pickup_datetime"))
        .withColumn("trip_duration_min",
            (unix_timestamp("tpep_dropoff_datetime") -
             unix_timestamp("tpep_pickup_datetime")) / 60.0)
        .filter(col("trip_duration_min").between(0.5, 240))
        .withColumn("label",
            when(col("tip_amount") / col("fare_amount") > 0.20, 1).otherwise(0))
        .select(
            "trip_distance", "fare_amount", "passenger_count",
            "pickup_hour", "pickup_dow", "pickup_month",
            col("PULocationID").cast("double").alias("pickup_zone"),
            col("payment_type").cast("double"),
            "trip_duration_min",
            "label",
            "tpep_pickup_datetime"
        )
        .dropna()
    )

df_baseline   = prepare_features(df_train_raw)
df_production = prepare_features(df_production_raw)

# 1% sample — sufficient for meaningful ML and drift analysis
# Kept small to avoid serverless session timeouts on Free Edition
train_full, test_all      = df_baseline.randomSplit([0.8, 0.2], seed=42)
train_df                  = train_full.sample(fraction=0.01, seed=42)
test_df                   = test_all.sample(fraction=0.01, seed=42)
df_production_sample      = df_production.sample(fraction=0.01, seed=42)

print(f"Training rows (sample):   {train_df.count():,}")
print(f"Test rows (sample):       {test_df.count():,}")
print(f"Production rows (sample): {df_production_sample.count():,}")

print("\nLabel balance (training — check for imbalance):")
train_df.groupBy("label").count().orderBy("label").show()
```

**Questions**

> **Q1.1** What does label = 1 represent in our model? Why might a taxi platform want to predict this, and what decisions could be informed by it?

> **Q1.2** Why do we use January (winter) as baseline and June (summer) as the production drift period? What seasonal and behavioural factors would shift the tip distribution between these months?

> **Q1.3** We sample 1% of the data. What are the trade-offs of this decision for (a) model quality, (b) drift detection sensitivity, and (c) runtime?

## 2. MLflow Experiment Tracking

#### Step 1 — Set up MLflow

```python
import mlflow

# Use your personal workspace path — replace the email with your own
# To find your path, run: import os; os.getcwd()
# It will return something like /Workspace/Users/your-email@domain.com
# Use the email portion below — do NOT include /Workspace in the MLflow path
mlflow.set_experiment("/Users/your-email@domain.com/com6012-lab10-taxi-tip")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
```

> **Note:** MLflow will print a message saying the experiment does not exist and is being created — this is expected on the first run.

#### Step 2 — Session Checkpoint

> **If you see a `SESSION_NOT_FOUND`, `INVALID_CONNECT_URL`, or `NO_ACTIVE_SESSION` error**, the serverless session has reset completely and all variables are lost. The only recovery is to detach and reattach Serverless compute from the notebook toolbar, then click **Run All** to re-execute the entire notebook from the top — including the data download. The checkpoint cell below is only useful if `df_train_raw` still exists in memory (i.e. a partial timeout). Before running it, verify with: `print(df_train_raw.count())`. If that raises a `NameError`, use Run All instead.

```python
# Checkpoint cell — only run this if df_train_raw is still defined
# If df_train_raw is not defined, use Run All from the top instead
# Verify first: print(df_train_raw.count())

from pyspark.sql.functions import col, hour, dayofweek, month, when, unix_timestamp

df_baseline   = prepare_features(df_train_raw)
df_production = prepare_features(df_production_raw)

train_full, test_all      = df_baseline.randomSplit([0.8, 0.2], seed=42)
train_df                  = train_full.sample(fraction=0.01, seed=42)
test_df                   = test_all.sample(fraction=0.01, seed=42)
df_production_sample      = df_production.sample(fraction=0.01, seed=42)

print("DataFrames rebuilt successfully.")
print(f"Training rows: {train_df.count():,}")
```

#### Step 3 — Train Random Forest

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

# mlflow.spark.autolog() is not supported on Free Edition serverless
# Parameters and metrics are logged manually below

auc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
acc_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
f1_eval  = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

with mlflow.start_run(run_name="random-forest-baseline") as run:
    mlflow.set_tag("model_type",  "random_forest")
    mlflow.set_tag("data_period", "jan-2019")

    mlflow.log_param("num_trees",        10)
    mlflow.log_param("max_depth",         4)
    mlflow.log_param("sample_fraction", 0.01)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=10, maxDepth=4, seed=42)
    pipeline = Pipeline(stages=[assembler, rf])
    model    = pipeline.fit(train_df)
    preds    = model.transform(test_df)

    auc = auc_eval.evaluate(preds)
    acc = acc_eval.evaluate(preds)
    f1  = f1_eval.evaluate(preds)

    mlflow.log_metric("test_auc",      auc)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1",       f1)

    mlflow.spark.log_model(model, "model")

    print(f"AUC: {auc:.4f}  |  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
    rf_run_id = run.info.run_id
```

#### Step 4 — Train GBT for Comparison

```python
from pyspark.ml.classification import GBTClassifier

with mlflow.start_run(run_name="gbt-baseline") as run:
    mlflow.set_tag("model_type", "gradient_boosted_trees")

    mlflow.log_param("max_iter",          5)
    mlflow.log_param("max_depth",         4)
    mlflow.log_param("sample_fraction", 0.01)

    assembler_g = VectorAssembler(inputCols=feature_cols, outputCol="features")
    gbt = GBTClassifier(
        featuresCol="features", labelCol="label",
        maxIter=5, maxDepth=4, seed=42)
    model_gbt = Pipeline(stages=[assembler_g, gbt]).fit(train_df)
    preds_gbt = model_gbt.transform(test_df)

    auc_gbt = auc_eval.evaluate(preds_gbt)
    acc_gbt = acc_eval.evaluate(preds_gbt)
    f1_gbt  = f1_eval.evaluate(preds_gbt)

    mlflow.log_metric("test_auc",      auc_gbt)
    mlflow.log_metric("test_accuracy", acc_gbt)
    mlflow.log_metric("test_f1",       f1_gbt)

    mlflow.spark.log_model(model_gbt, "model")

    print(f"GBT — AUC: {auc_gbt:.4f}  |  Accuracy: {acc_gbt:.4f}  |  F1: {f1_gbt:.4f}")
    gbt_run_id = run.info.run_id
```

#### Step 5 — Compare in the MLflow UI

In the Databricks left sidebar, click **Experiments**, find your experiment, select both runs, then click **Compare**. Explore the Parallel Coordinates chart and metric comparison table.

**Questions**

> **Q2.1** Why is **AUC** more appropriate than Accuracy as the primary metric for this task? Refer to the label balance you observed in Section 1.

> **Q2.2** In this lab we log parameters and metrics manually. What would `mlflow.spark.autolog()` have captured automatically that we have not logged here? List at least three things.

> **Q2.3** What is the conceptual difference between a **tag** and a **parameter** in MLflow? When would you use each?

## 3. Model Registration and Batch Inference

#### Step 1 — Register to Unity Catalog

```python
from mlflow import MlflowClient
client = MlflowClient()

best_run_id = rf_run_id   # change to gbt_run_id if GBT performed better

# Unity Catalog format: catalog.schema.model_name
model_name = "ml.com6012.taxi_tip_classifier_abc1xy"   # replace abc1xy

registered = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model", name=model_name)
print(f"Registered: {registered.name}  v{registered.version}")
```

> **Free Edition note.** If the schema `ml.com6012` does not exist, run:
> ```python
> spark.sql("CREATE SCHEMA IF NOT EXISTS ml.com6012")
> ```
> Or use a personal schema: `"[your_username].default.taxi_tip_classifier_abc1xy"`

#### Step 2 — Add Alias and Description

```python
client.update_model_version(
    name=model_name, version=registered.version,
    description="RF classifier for NYC taxi tip prediction (Jan 2019). Predicts tip > 20% of fare.")
client.set_registered_model_alias(
    name=model_name, alias="champion", version=registered.version)
print(f"Model v{registered.version} aliased as 'champion'")
```

#### Step 3 — Load from Registry and Score Production Data

```python
loaded_model = mlflow.spark.load_model(f"models:/{model_name}@champion")
preds_prod   = loaded_model.transform(
    df_production_sample.select(feature_cols + ["label"]))
auc_prod     = auc_eval.evaluate(preds_prod)

print(f"AUC — winter test set:        {auc:.4f}")
print(f"AUC — summer production data: {auc_prod:.4f}")
print(f"AUC drop:                     {auc - auc_prod:.4f}")
```

**Questions**

> **Q3.1** How large is the AUC drop between the winter test set and the summer production data? Does this surprise you given the seasonal context?

> **Q3.2** What is the purpose of the `champion` alias, and why does it replace the old `Staging`/`Production` stage system in modern MLflow?

## 4. Drift Detection: Statistics, Visualisation, and Interpretable Metrics

#### Step 1 — Convert to Pandas for Analysis

```python
import pandas as pd, numpy as np
from scipy import stats

pdf_baseline   = train_df.select(feature_cols).toPandas()
pdf_production = df_production_sample.select(feature_cols).toPandas()
print(f"Baseline rows:   {len(pdf_baseline):,}")
print(f"Production rows: {len(pdf_production):,}")
```

#### Step 2 — Visualise Feature Distributions

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features_to_plot = [
    "trip_distance", "fare_amount", "trip_duration_min",
    "pickup_hour",   "pickup_dow",  "pickup_month"
]
for ax, feat in zip(axes.flatten(), features_to_plot):
    ax.hist(pdf_baseline[feat],   bins=30, alpha=0.6,
            label="Baseline (Jan)", color="steelblue",  density=True)
    ax.hist(pdf_production[feat], bins=30, alpha=0.6,
            label="Production (Jun)", color="darkorange", density=True)
    ax.set_title(feat); ax.legend(fontsize=8)
plt.suptitle("Feature Distributions: Baseline vs. Production", fontsize=14)
plt.tight_layout()
plt.savefig("/tmp/drift_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
```

#### Step 3 — KS Test

```python
continuous_features = [
    "trip_distance", "fare_amount", "trip_duration_min", "pickup_hour"
]
print("=" * 62)
print(f"{'Feature':<25} {'KS Stat':>10} {'p-value':>12} {'Drift?':>8}")
print("=" * 62)
drift_results = {}
for feat in continuous_features:
    ks, pv = stats.ks_2samp(
        pdf_baseline[feat].dropna(), pdf_production[feat].dropna())
    drift_results[feat] = {"ks_stat": ks, "p_value": pv}
    print(f"{feat:<25} {ks:>10.4f} {pv:>12.6f} {'YES' if pv < 0.05 else 'no':>8}")
```

#### Step 4 — Population Stability Index (PSI)

```python
def compute_psi(baseline, production, feature, n_bins=10):
    """PSI: < 0.10 stable | 0.10–0.20 investigate | > 0.20 action required"""
    _, edges = np.histogram(baseline[feature].dropna(), bins=n_bins)
    edges[0] = -np.inf; edges[-1] = np.inf
    b = np.histogram(baseline[feature].dropna(),   bins=edges)[0] / len(baseline)
    p = np.histogram(production[feature].dropna(), bins=edges)[0] / len(production)
    b = np.where(b == 0, 0.0001, b); p = np.where(p == 0, 0.0001, p)
    return float(np.sum((p - b) * np.log(p / b)))

print("\nPSI Results")
print("=" * 47)
for feat in continuous_features:
    psi = compute_psi(pdf_baseline, pdf_production, feat)
    status = "Stable" if psi < 0.10 else ("Warning" if psi < 0.20 else "DRIFT")
    print(f"{feat:<25} {psi:>8.4f}  {status}")
```

#### Step 5 — Timestamp-Indexed Accuracy: Predictions vs. Actuals Over Time

Rather than comparing aggregate statistics, we compare model accuracy on the two time periods as time series. This reveals whether drift is gradual or sudden.

```python
import matplotlib.dates as mdates

def score_with_timestamps(fitted_model, df):
    return (
        fitted_model
        .transform(df.select(feature_cols + ["label", "tpep_pickup_datetime"]))
        .select("tpep_pickup_datetime", "prediction", "label")
        .toPandas()
        .assign(datetime=lambda x: pd.to_datetime(x["tpep_pickup_datetime"]))
        .sort_values("datetime")
    )

preds_base_pdf = score_with_timestamps(model, test_df)
preds_prod_pdf = score_with_timestamps(model, df_production_sample)

def rolling_accuracy(df, window="6H"):
    df2 = df.set_index("datetime")
    df2["correct"] = (df2["prediction"] == df2["label"]).astype(int)
    return df2["correct"].resample(window).mean().reset_index()

acc_base = rolling_accuracy(preds_base_pdf)
acc_prod = rolling_accuracy(preds_prod_pdf)

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

axes[0].plot(acc_base["datetime"], acc_base["correct"],
             color="steelblue", linewidth=1.5, label="Baseline (Jan)")
axes[0].axhline(acc_base["correct"].mean(), color="steelblue",
                linestyle="--", alpha=0.6,
                label=f"Mean: {acc_base['correct'].mean():.3f}")
axes[0].set_title("Rolling 6-hour Accuracy — Baseline Period")
axes[0].set_xlabel("Date"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

baseline_mean = acc_base["correct"].mean()
axes[1].plot(acc_prod["datetime"], acc_prod["correct"],
             color="darkorange", linewidth=1.5, label="Production (Jun)")
axes[1].axhline(acc_prod["correct"].mean(), color="darkorange",
                linestyle="--", alpha=0.6,
                label=f"Production mean: {acc_prod['correct'].mean():.3f}")
axes[1].axhline(baseline_mean, color="steelblue",
                linestyle=":", alpha=0.7,
                label=f"Baseline mean: {baseline_mean:.3f}")
axes[1].set_title("Rolling 6-hour Accuracy — Production Period")
axes[1].set_xlabel("Date"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

plt.suptitle("Model Accuracy Over Time: Baseline vs. Production", fontsize=13)
plt.tight_layout()
plt.savefig("/tmp/accuracy_over_time.png", dpi=120, bbox_inches="tight")
plt.show()
```

#### Step 6 — % Predictions Within Acceptable Range

KS statistics are statistically rigorous but difficult for non-technical stakeholders to act on. We complement them with a business-interpretable indicator: the percentage of time windows in which model accuracy falls within an acceptable tolerance band around the baseline mean.

```python
tolerance = 0.05   # ±5 percentage points

prod_in_range = (
    acc_prod["correct"].between(baseline_mean - tolerance, baseline_mean + tolerance)
).mean() * 100

base_in_range = (
    acc_base["correct"].between(baseline_mean - tolerance, baseline_mean + tolerance)
).mean() * 100

print(f"Baseline mean accuracy:              {baseline_mean:.4f}")
print(f"Acceptable band (±{tolerance*100:.0f}pp):           "
      f"[{baseline_mean - tolerance:.4f}, {baseline_mean + tolerance:.4f}]")
print(f"\nBaseline windows within range:       {base_in_range:.1f}%")
print(f"Production windows within range:     {prod_in_range:.1f}%")

if prod_in_range < 60:
    print("\nDRIFT ALERT: fewer than 60% of production windows within acceptable range.")
    print("   Recommended action: investigate feature distributions, trigger retraining.")
else:
    print("\nProduction performance broadly within acceptable range.")
```

#### Step 7 — Log All Drift Metrics to MLflow

```python
with mlflow.start_run(run_name="drift-analysis-jan-vs-jun"):
    mlflow.set_tag("analysis_type",     "drift_detection")
    mlflow.set_tag("baseline_period",   "jan-2019")
    mlflow.set_tag("production_period", "jun-2019")

    for feat in continuous_features:
        r = drift_results[feat]
        mlflow.log_metric(f"ks_stat_{feat}", r["ks_stat"])
        mlflow.log_metric(f"ks_pval_{feat}", r["p_value"])
        mlflow.log_metric(f"psi_{feat}",     compute_psi(pdf_baseline, pdf_production, feat))

    mlflow.log_metric("pct_prod_windows_in_range", prod_in_range)
    mlflow.log_metric("pct_base_windows_in_range", base_in_range)
    mlflow.log_metric("auc_baseline",   auc)
    mlflow.log_metric("auc_production", auc_prod)
    mlflow.log_metric("auc_delta",      auc - auc_prod)

    mlflow.log_artifact("/tmp/drift_distributions.png")
    mlflow.log_artifact("/tmp/accuracy_over_time.png")
    print("Drift metrics and plots logged to MLflow.")
```

**Questions**

> **Q4.1** Which features showed the highest drift (KS statistic, PSI)? Does this make intuitive sense given the January vs. June comparison? Explain the physical or behavioural factors driving each feature's shift.

> **Q4.2** We apply the KS test to four features simultaneously. What problem does this create statistically, and how would you address it? (Hint: Bonferroni correction.)

> **Q4.3** The `% predictions within acceptable range` metric uses ±5 percentage points as the tolerance. Is this threshold arbitrary? How might you set it more rigorously using only the baseline data?

> **Q4.4** The timestamp-indexed accuracy plot shows whether drift is gradual or abrupt. What does your plot reveal? Does accuracy drop uniformly across June, or are there specific patterns that stand out?

## 5. Incremental Inference Simulation and Lakehouse Monitoring

In Sections 3 and 4, we scored all production data at once. In reality, inference logs accumulate gradually — each serving request adds one row. This section simulates that pattern, producing a more realistic inference table for monitoring.

> **Free Edition note.** The Lakehouse Monitoring SDK in Step 5 requires a paid workspace. Complete Steps 1–4 regardless — they demonstrate the incremental pattern and produce the plots needed to answer the questions.

#### Step 1 — Prepare Time-Ordered Production Data

```python
from pyspark.sql.functions import lit, monotonically_increasing_id
from functools import reduce

prod_sorted = (
    df_production_sample
    .select(feature_cols + ["label", "tpep_pickup_datetime"])
    .orderBy("tpep_pickup_datetime")
    .cache()
)
total_rows = prod_sorted.count()
BATCH_SIZE = max(1, total_rows // 10)
print(f"Total rows: {total_rows:,}  |  Batch size: {BATCH_SIZE:,}  |  Batches: 10")
```

#### Step 2 — Incremental Scoring Loop

```python
all_batches = []

for batch_num in range(10):
    start = batch_num * BATCH_SIZE
    end   = min(start + BATCH_SIZE, total_rows)

    batch_df = prod_sorted.limit(end).subtract(prod_sorted.limit(start))

    scored = (
        model.transform(batch_df.select(feature_cols + ["label", "tpep_pickup_datetime"]))
        .withColumn("batch_number",  lit(batch_num + 1))
        .withColumn("model_version", lit("1"))
        .withColumn("request_id",    monotonically_increasing_id())
        .select("request_id", "tpep_pickup_datetime", "model_version",
                "batch_number", *feature_cols, "prediction", "label")
    )
    all_batches.append(scored)
    max_ts = scored.agg({"tpep_pickup_datetime": "max"}).collect()[0][0]
    print(f"  Batch {batch_num+1:2d}/10: {scored.count():,} rows scored  (up to {max_ts})")

print("\nAll batches complete.")
```

#### Step 3 — Write Inference Log to Delta Table

```python
inference_log = (
    reduce(lambda a, b: a.union(b), all_batches)
    .withColumnRenamed("tpep_pickup_datetime", "request_timestamp")
)

inference_table_path = "ml.com6012.taxi_tip_inference_log_abc1xy"   # replace abc1xy
(inference_log.write.format("delta")
 .mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(inference_table_path))
print(f"Inference log: {inference_log.count():,} rows → {inference_table_path}")
```

#### Step 4 — Visualise Accumulating Accuracy

```python
pdf_log = spark.table(inference_table_path).toPandas()
pdf_log["request_timestamp"] = pd.to_datetime(pdf_log["request_timestamp"])
pdf_log = pdf_log.sort_values("request_timestamp")

batch_metrics = []
for b in range(1, 11):
    subset = pdf_log[pdf_log["batch_number"] <= b]
    batch_metrics.append({
        "batch": b,
        "cumulative_accuracy": (subset["prediction"] == subset["label"]).mean(),
        "up_to": subset["request_timestamp"].max()
    })
pdf_metrics = pd.DataFrame(batch_metrics)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(pdf_metrics["up_to"], pdf_metrics["cumulative_accuracy"],
        marker="o", color="darkorange", linewidth=2, markersize=8,
        label="Cumulative accuracy (production)")
ax.axhline(y=baseline_mean, color="steelblue",
           linestyle="--", linewidth=1.5, label=f"Baseline: {baseline_mean:.3f}")
ax.fill_between(
    pdf_metrics["up_to"],
    baseline_mean - tolerance, baseline_mean + tolerance,
    alpha=0.12, color="steelblue", label=f"Acceptable range (±{int(tolerance*100)}pp)"
)
ax.set_xlabel("Date (as inference accumulates)")
ax.set_ylabel("Cumulative Accuracy")
ax.set_title("Accuracy as Production Inference Accumulates Over Time")
ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.tight_layout()
plt.savefig("/tmp/incremental_inference.png", dpi=120, bbox_inches="tight")
plt.show()
```

#### Step 5 — Create Lakehouse Monitor (Paid/Trial Workspaces Only)

```python
baseline_table_path = "ml.com6012.taxi_tip_training_baseline_abc1xy"
(train_df.select(feature_cols + ["label"]).withColumn("model_version", lit("1"))
 .write.format("delta").mode("overwrite").saveAsTable(baseline_table_path))

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog, MonitorInferenceLogProblemType)
w = WorkspaceClient()
try:
    m = w.quality_monitors.create(
        table_name=inference_table_path,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction", label_col="label",
            model_id_col="model_version", timestamp_col="request_timestamp"),
        baseline_table_name=baseline_table_path,
        slicing_exprs=["batch_number"],
        output_schema_name="ml.com6012",
    )
    print(f"Monitor created!\nProfile: {m.profile_metrics_table_name}")
except Exception as e:
    print(f"Monitor API not available (expected on Free Edition): {e}")
    print("The incremental inference plots above serve the equivalent purpose.")
```

**Questions**

> **Q5.1** Look at the cumulative accuracy plot. Does accuracy degrade uniformly as more batches arrive, or is there a specific point where it drops? What might cause a sudden vs. gradual degradation?

> **Q5.2** Compare the incremental approach (this section) with one-shot batch scoring (Section 4). What does the incremental approach reveal that the one-shot approach conceals? In a real production system, which approach reflects reality?

> **Q5.3** The ±5 percentage point tolerance is fixed. What would be a more principled approach to setting this threshold — for instance, using the variance of the baseline rolling accuracy itself?

## 6. Retraining on Combined Data

When drift is detected, the standard response is to retrain on more recent data.

```python
df_combined     = train_df.union(df_production_sample.sample(fraction=0.8, seed=42))
train_c, test_c = df_combined.randomSplit([0.8, 0.2], seed=42)

with mlflow.start_run(run_name="rf-retrained-combined") as retrain_run:
    mlflow.set_tag("trigger",     "drift_detected")
    mlflow.set_tag("data_period", "jan + jun 2019")
    mlflow.log_param("num_trees", 10)
    mlflow.log_param("max_depth",  4)

    model_r = Pipeline(stages=[
        VectorAssembler(inputCols=feature_cols, outputCol="features"),
        RandomForestClassifier(featuresCol="features", labelCol="label",
                               numTrees=10, maxDepth=4, seed=42)
    ]).fit(train_c)

    auc_r      = auc_eval.evaluate(model_r.transform(test_c))
    auc_prod_r = auc_eval.evaluate(
        model_r.transform(df_production_sample.select(feature_cols + ["label"])))
    mlflow.log_metric("auc_combined_test", auc_r)
    mlflow.log_metric("auc_production",    auc_prod_r)

    print(f"Retrained AUC (combined test):     {auc_r:.4f}")
    print(f"Retrained AUC (summer production): {auc_prod_r:.4f}")
    print(f"Original AUC (summer production):  {auc_prod:.4f}")
    print(f"Improvement:                       {auc_prod_r - auc_prod:.4f}")

v = mlflow.register_model(f"runs:/{retrain_run.info.run_id}/model", model_name)
client.set_registered_model_alias(name=model_name, alias="challenger", version=v.version)
print(f"\nRetrained model registered as v{v.version}, alias: 'challenger'")
```

**Questions**

> **Q6.1** Did retraining on combined data improve performance on the summer production set? Why might combining both periods not always be optimal — what are the risks?

> **Q6.2** Describe a safe promotion strategy for moving Challenger to Champion. What tests and approvals would you require before updating the alias?

## Appendix A — Troubleshooting

**Free Edition compute quota exceeded.** Compute resumes tomorrow. Data and notebooks are preserved. Reduce the sample fraction to `0.005` if needed.

**Session timeout (SESSION_NOT_FOUND, INVALID_CONNECT_URL, or NO_ACTIVE_SESSION).** The serverless session has reset completely and all variables are lost. Detach and reattach Serverless compute from the notebook toolbar, then click **Run All** to re-execute the entire notebook from the top. The data will need to be re-downloaded. The checkpoint cell in Section 2 Step 2 can only help if `df_train_raw` is still in memory — verify with `print(df_train_raw.count())` before using it.

**SparkSession error (INVALID_CONNECT_URL).** Do not call `SparkSession.builder.getOrCreate()` in Databricks notebooks. The `spark` variable is pre-provided automatically in every notebook session.

**Public TLC URL slow or unavailable.** Try loading a single month:

    df = spark.createDataFrame(
        pd.read_parquet(
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet"
        )
    )

**MLflow experiment path.** To find your correct path run `import os; os.getcwd()`. Use the email portion of the output as the base: `/Users/your-email@domain.com/experiment-name`. Do not include `/Workspace` in the MLflow path.

**`mlflow.spark.autolog()` error.** This is not supported on Free Edition serverless clusters. The lab uses manual logging instead — remove any `mlflow.spark.autolog()` calls if encountered.

**Unity Catalog schema not found.**

    spark.sql("CREATE CATALOG IF NOT EXISTS ml")
    spark.sql("CREATE SCHEMA IF NOT EXISTS ml.com6012")

**`databricks.sdk` import error.**

    %pip install databricks-sdk --upgrade
    dbutils.library.restartPython()

## Appendix B — Dataset Quick Reference

| Field | Used as | Notes |
|---|---|---|
| `tpep_pickup_datetime` | Temporal split; timestamps for drift plots | Key for seasonal comparison |
| `trip_distance` | Feature | Miles; shifts seasonally |
| `fare_amount` | Feature | Base fare; correlated with distance |
| `tip_amount` | Label source | `tip/fare > 0.20` → label = 1 |
| `passenger_count` | Feature | Integer |
| `PULocationID` | Feature (as `pickup_zone`) | NYC taxi zone ID |
| `payment_type` | Feature | 2 = cash → tip always 0 |
| `tpep_dropoff_datetime` | Compute `trip_duration_min` | Not used directly as feature |

## References

- NYC TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Databricks Free Edition: https://signup.databricks.com
- Free Edition limitations: https://docs.databricks.com/aws/en/getting-started/free-edition-limitations
- Educator guide (CE → Free Edition): https://community.databricks.com/t5/databricks-university-alliance/guide-and-best-practices-moving-from-community-edition-to-free/ta-p/129308
- MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
- Databricks Data Quality Monitoring: https://docs.databricks.com/aws/en/data-quality-monitoring/
- Marvelous MLOps end-to-end pipeline: https://github.com/marvelousmlops/marvel-characters
- Vechtomova, M. (2025). *MLOps with Databricks: Machine Learning End-to-End*. O'Reilly Media. https://www.oreilly.com/library/view/mlops-with-databricks/9798341608245/
- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly Media. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/

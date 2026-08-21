"""
Model inference (Phase 5).

Loads a trained model from the local model registry and generates 72-hour
AQI forecasts from new (or held-out) feature data.

The registry stores, alongside every model, the exact ordered feature list
it was trained on (`feature_columns.json`). Inference simply:
    - picks a run directory (`--run-dir` or the most recent one),
    - loads the fitted model + its feature columns,
    - reorders/pads the input data to exactly those columns,
    - calls `model.predict(...)`.

Because every production model is a tree-based ensemble, no feature scaling
is required at inference time (the pipeline never persists a scaler).

Run directly with:
    python -m src.training_pipeline.predict --input path/to/features.csv
    python -m src.training_pipeline.predict                          # uses data/processed/feature_df.csv
    python -m src.training_pipeline.predict --run-dir data/model_registry/lightgbm_20260101T000000Z
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs.config import PROJECT_ROOT
from src.common.exceptions import AQIPipelineError, TrainingError
from src.common.logger import get_logger
from src.training_pipeline.data_loader import DEFAULT_FEATURE_DF_PATH, load_features_from_local_csv
from src.training_pipeline.model_registry import find_latest_run, load_model_locally

logger = get_logger(__name__)


def build_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Build the exact feature matrix a registry model expects.

    Selects, in order, the columns listed in the model's
    `feature_columns.json`, raising if any required feature is missing. Rows
    with missing feature values are dropped (the model cannot make a
    meaningful forecast without them).

    Args:
        df: Raw input data (typically the engineered feature dataframe).
        feature_columns: Ordered feature list from the model's metadata.

    Returns:
        DataFrame with exactly `feature_columns` as its columns, in order,
        with any rows containing NaN feature values removed.

    Raises:
        TrainingError: If any required feature column is absent from `df`.
    """
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise TrainingError(
            "Input data is missing required feature columns: "
            f"{missing}. Re-run feature engineering to regenerate them."
        )

    X = df[feature_columns].copy()
    dropped = X.isna().any(axis=1).sum()
    if dropped:
        logger.warning("Dropped %d row(s) with missing feature values.", dropped)
        X = X.dropna()
    return X


def load_predictor(
    run_dir: Path | str | None,
) -> tuple[Any, list[str], dict]:
    """Load a trained model, its feature columns, and metadata from the registry.

    Args:
        run_dir: Explicit registry run directory, or `None` to use the most
            recent run found by `find_latest_run`.

    Returns:
        Tuple of `(model, feature_columns, metadata)`.

    Raises:
        TrainingError: If no run directory can be resolved or the load fails.
    """
    if run_dir is None:
        resolved = find_latest_run()
        if resolved is None:
            raise TrainingError(
                "No trained model found in the local registry. Train one first: "
                "python -m src.training_pipeline.train_pipeline --source local"
            )
        run_dir = resolved

    model, metadata = load_model_locally(run_dir)
    feature_columns = list(metadata["feature_columns"])
    return model, feature_columns, metadata


def predict(
    df: pd.DataFrame,
    run_dir: Path | str | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Generate forecasts for `df` using a registry model.

    Args:
        df: Engineered feature data (must contain the model's feature
            columns).
        run_dir: Registry run directory to use, or `None` for the most
            recent run.

    Returns:
        Tuple of `(predictions, feature_columns, metadata)` where
        `predictions` is a 1-D array of 72h-ahead AQI forecasts aligned with
        the non-dropped rows of `df`.
    """
    model, feature_columns, metadata = load_predictor(run_dir)
    X = build_feature_matrix(df, feature_columns)
    y_pred = np.asarray(model.predict(X), dtype="float64")
    logger.info(
        "Generated %d forecast(s) with model '%s' using %d feature(s).",
        len(y_pred),
        metadata.get("model_name", "unknown"),
        len(feature_columns),
    )
    return y_pred, feature_columns, metadata


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = success, 1 = failure)."""
    parser = argparse.ArgumentParser(description="Generate AQI forecasts from a trained model.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to the engineered feature CSV to forecast on "
            "(default: data/processed/feature_df.csv)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Model registry run directory to load (default: the most recent "
            "run found in the local registry)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write predictions as CSV (default: print to stdout).",
    )
    args = parser.parse_args()

    try:
        input_path = Path(args.input) if args.input else DEFAULT_FEATURE_DF_PATH
        df = load_features_from_local_csv(input_path)

        predictions, feature_columns, metadata = predict(df, run_dir=args.run_dir)
        model_name = metadata.get("model_name", "unknown")

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"aqi_forecast_72h": predictions}).to_csv(out, index=False)
            print(f"Wrote {len(predictions)} forecast(s) to: {out}")
        else:
            print(f"Model: {model_name}")
            print(f"Features: {len(feature_columns)}")
            print(f"Forecasts (72h-ahead AQI): {predictions.tolist()}")
        return 0
    except AQIPipelineError as exc:
        logger.error("Prediction failed: %s", exc)
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Prediction failed with an unexpected error")
        print(f"Prediction failed with an unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

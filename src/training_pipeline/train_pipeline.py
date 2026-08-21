"""
Model training pipeline (Phase 5).

Workflow:

    1. Load engineered features + target from the Feature Store (Hopsworks,
       default) or a local CSV (`--source local`, for fast iteration).
    2. Select the fixed `FEATURE_COLUMNS` whitelist (see `preprocessing.py`).
    3. Split chronologically into train / validation / test (70 / 15 / 15 by
       default). Data is never shuffled.
    4. Tune each tree-based candidate (LightGBM, XGBoost, CatBoost, Extra
       Trees, Random Forest) with `RandomizedSearchCV` over a
       `TimeSeriesSplit` cross-validator whose `gap` equals the 72-hour
       forecast horizon, preventing temporal leakage during tuning.
    5. Evaluate every tuned candidate (plus the Persistence Baseline, which
       predicts the 72h-ahead AQI using the current AQI) on the validation
       set: RMSE, MAE, R², sMAPE.
    6. Select the best model by lowest validation RMSE (the persistence
       baseline is compared but never selected as the production model).
    7. Retrain the selected model on train+validation, evaluate it on the
       untouched test set, and save it -- plus `feature_columns.json` and a
       `metadata.json` (model name, training date, feature list, split
       sizes, metrics, package versions) -- to the local model registry.
       Optionally (best-effort) also register it in the Hopsworks Model
       Registry.

Run directly with:
    python -m src.training_pipeline.train_pipeline
    python -m src.training_pipeline.train_pipeline --source local
    python -m src.training_pipeline.train_pipeline --train-size 0.7 --val-size 0.15 --test-size 0.15
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from src.common.exceptions import AQIPipelineError, TrainingError
from src.common.logger import get_logger
from src.training_pipeline.data_loader import DataSource, load_features
from src.training_pipeline.evaluate import (
    build_comparison_table,
    evaluate_model,
    select_best_model,
)
from src.training_pipeline.model_registry import (
    register_model_to_hopsworks,
    save_model_locally,
)
from src.training_pipeline.models import (
    PERSISTENCE_MODEL_NAME,
    PersistenceBaselineRegressor,
    get_model_candidates,
    get_param_distributions,
    get_package_versions,
)
from src.training_pipeline.preprocessing import (
    TARGET_COLUMN,
    build_xy,
    make_time_series_cv,
    select_feature_columns,
    time_based_train_val_test_split,
)
from src.training_pipeline.walk_forward import (
    run_walk_forward,
    run_walk_forward_persistence,
)

logger = get_logger(__name__)

DEFAULT_N_SPLITS = 5
DEFAULT_GAP = 72  # forecast horizon in samples (72 hours of hourly data)
DEFAULT_N_ITER = 25
DEFAULT_WALK_FORWARD_STEP = 336  # 2 weeks of hourly data per walk-forward block
DEFAULT_WALK_FORWARD_MIN_TRAIN = 1500


def tune_model(
    base_model: Any,
    param_distributions: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_splits: int = DEFAULT_N_SPLITS,
    gap: int = DEFAULT_GAP,
    n_iter: int = DEFAULT_N_ITER,
) -> Any:
    """Tune a single model family using chronological cross-validation.

    Args:
        base_model: An unfitted estimator from `get_model_candidates`.
        param_distributions: Hyperparameter search space from
            `get_param_distributions`. An empty dict means the model has no
            tunable parameters; it is fitted once on `X_train` and returned.
        X_train: Training feature matrix.
        y_train: Training target vector.
        random_state: Seed for the search (reproducible sampling).
        n_splits: Number of `TimeSeriesSplit` folds.
        gap: Samples left between each train/test fold (default 72 = the
            forecast horizon) to prevent temporal leakage.
        n_iter: Number of `RandomizedSearchCV` parameter samples.

    Returns:
        The best estimator found by the search, already refit on the full
        `X_train` (or, for parameter-free models, fitted once on
        `X_train`).

    Raises:
        TrainingError: If `X_train` is too small for the requested CV.
    """
    if not param_distributions:
        # No tunable parameters (e.g. future parameter-free baselines):
        # fit once and return.
        model = base_model
        model.fit(X_train, y_train.to_numpy())
        return model

    # TimeSeriesSplit needs enough samples to form `n_splits` folds plus the
    # gap between each; shrink the split count gracefully on small datasets.
    n = len(X_train)
    while n_splits > 2 and n < (n_splits + 1) + gap * n_splits:
        n_splits -= 1
    cv = make_time_series_cv(n_splits=n_splits, gap=gap)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    try:
        search.fit(X_train, y_train.to_numpy())
    except ValueError as exc:
        raise TrainingError(
            f"Hyperparameter search failed for model '{base_model.__class__.__name__}': {exc}"
        ) from exc

    logger.info(
        "Tuned '%s': best CV RMSE=%.3f (%d search iterations, %d splits, gap=%d)",
        base_model.__class__.__name__,
        -search.best_score_,
        n_iter,
        n_splits,
        gap,
    )
    return search.best_estimator_


def run_training_pipeline(
    source: DataSource = "hopsworks",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    n_splits: int = DEFAULT_N_SPLITS,
    gap: int = DEFAULT_GAP,
    n_iter: int = DEFAULT_N_ITER,
    register_hopsworks: bool = False,
    local_path: Any = None,
    walk_forward: bool = True,
    walk_forward_step: int = DEFAULT_WALK_FORWARD_STEP,
    walk_forward_min_train: int = DEFAULT_WALK_FORWARD_MIN_TRAIN,
) -> dict[str, Any]:
    """Execute the full training pipeline end-to-end, once.

    Args:
        source: `"hopsworks"` (default) or `"local"` -- see `data_loader.py`.
        train_size: Fraction of rows (chronologically, earliest) for training.
        val_size: Fraction of rows (chronologically, middle) for validation.
        test_size: Fraction of rows (chronologically, most recent) for final
            testing.
        random_state: Shared random seed for reproducibility.
        n_splits: Number of `TimeSeriesSplit` folds used for tuning.
        gap: Forecast-horizon gap (in samples) between CV folds.
        n_iter: Number of `RandomizedSearchCV` parameter samples per model.
        register_hopsworks: If `True`, also attempt to push the best model
            to the Hopsworks Model Registry (best-effort; see
            `model_registry.register_model_to_hopsworks`).
        local_path: Path to `feature_df.csv`, used only when
            `source="local"`. Defaults to
            `data_loader.DEFAULT_FEATURE_DF_PATH` if omitted.
        walk_forward: If `True`, additionally run expanding-window
            (walk-forward) backtesting of the best model over the whole
            dataset. Walk-forward is the honest measure of forecast skill
            for this dataset: a single holdout test window can land entirely
            inside one low-variance seasonal regime (see `walk_forward.py`).
        walk_forward_step: Rows per walk-forward validation block.
        walk_forward_min_train: Minimum rows in the first (expanding)
            walk-forward training window.

    Returns:
        A results dict with keys:
            - `"comparison_table"`: `pd.DataFrame` of validation metrics for
              all candidates (including the persistence baseline), sorted by
              RMSE.
            - `"best_model_name"`: name of the selected model.
            - `"best_model_metrics"`: that model's TEST-set metrics dict.
            - `"persistence_metrics"`: the persistence baseline's TEST-set
              metrics dict.
            - `"walk_forward_metrics"`: the best model's walk-forward metrics
              dict (or `None` if disabled/skipped).
            - `"walk_forward_persistence_metrics"`: the persistence
              baseline's walk-forward metrics dict (or `None`).
            - `"walk_forward_n"`: number of rows scored walk-forward.
            - `"split_sizes"`: row counts `{"train": .., "val": .., "test": ..}`
              after incomplete-row removal.
            - `"registry_path"`: local registry path the best model was saved to.
            - `"hopsworks_registered"`: bool, whether the Hopsworks push succeeded.

    Raises:
        TrainingError: If any stage fails (loading, preprocessing, training,
            evaluation, or saving).
    """
    logger.info("=" * 60)
    logger.info("Training pipeline started (source=%s)", source)

    from src.training_pipeline.data_loader import DEFAULT_FEATURE_DF_PATH

    df = load_features(source=source, local_path=local_path or DEFAULT_FEATURE_DF_PATH)

    feature_cols = select_feature_columns(df)
    train_df, val_df, test_df = time_based_train_val_test_split(
        df, train_size=train_size, val_size=val_size, test_size=test_size
    )

    X_train, y_train = build_xy(train_df, feature_cols, TARGET_COLUMN)
    X_val, y_val = build_xy(val_df, feature_cols, TARGET_COLUMN)
    X_test, y_test = build_xy(test_df, feature_cols, TARGET_COLUMN)
    logger.info(
        "Train: %d rows | Val: %d rows | Test: %d rows | %d features",
        len(X_train), len(X_val), len(X_test), len(feature_cols),
    )

    # 1) Tune each tree-based candidate on the training split only, using
    #    chronological cross-validation with a horizon-sized gap.
    candidates = get_model_candidates(random_state=random_state)
    tunable_names = [name for name in candidates if name != PERSISTENCE_MODEL_NAME]

    tuned_models: dict[str, Any] = {}
    for name in tunable_names:
        logger.info("Tuning model: %s", name)
        tuned_models[name] = tune_model(
            base_model=candidates[name],
            param_distributions=get_param_distributions(name),
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
            n_splits=n_splits,
            gap=gap,
            n_iter=n_iter,
        )

    # 2) Evaluate every tuned candidate on the validation split (never the
    #    test split -- the test set is only touched once, at the very end).
    val_results: dict[str, dict[str, float]] = {}
    for name in tunable_names:
        metrics = evaluate_model(tuned_models[name], X_val, y_val.to_numpy())
        logger.info(
            "%s (val) -- RMSE=%.3f MAE=%.3f R2=%.3f sMAPE=%.3f",
            name, metrics["rmse"], metrics["mae"], metrics["r2"], metrics["smape"],
        )
        val_results[name] = metrics

    # 3) Persistence baseline: predict the 72h-ahead AQI using the current
    #    AQI. Compared against every candidate, but never selected.
    persistence_model = PersistenceBaselineRegressor()
    persistence_model.fit(X_val, y_val.to_numpy())
    val_results[PERSISTENCE_MODEL_NAME] = evaluate_model(
        persistence_model, X_val, y_val.to_numpy()
    )

    # 4) Select the best production model by lowest validation RMSE,
    #    excluding the persistence baseline.
    selection_pool = {k: v for k, v in val_results.items() if k != PERSISTENCE_MODEL_NAME}
    best_model_name = select_best_model(selection_pool)
    best_model = tuned_models[best_model_name]

    # 5) Retrain the winner on train+validation and evaluate once on test.
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    best_model.fit(X_trainval, y_trainval.to_numpy())
    best_metrics = evaluate_model(best_model, X_test, y_test.to_numpy())

    persistence_test_model = PersistenceBaselineRegressor()
    persistence_test_model.fit(X_test, y_test.to_numpy())
    persistence_metrics = evaluate_model(
        persistence_test_model, X_test, y_test.to_numpy()
    )

    logger.info(
        "Best model: %s (test RMSE=%.3f MAE=%.3f R2=%.3f sMAPE=%.3f)",
        best_model_name, best_metrics["rmse"], best_metrics["mae"],
        best_metrics["r2"], best_metrics["smape"],
    )
    logger.info(
        "Persistence baseline (test): RMSE=%.3f MAE=%.3f R2=%.3f sMAPE=%.3f",
        persistence_metrics["rmse"], persistence_metrics["mae"],
        persistence_metrics["r2"], persistence_metrics["smape"],
    )

    # 6) Walk-forward backtest of the best model over the whole dataset --
    #    the honest measure of forecast skill when a single holdout window
    #    can land entirely inside one seasonal regime (see walk_forward.py).
    wf_metrics: dict[str, Any] | None = None
    wf_persistence_metrics: dict[str, Any] | None = None
    wf_n = 0
    if walk_forward:
        logger.info("Running walk-forward (expanding-window) evaluation of %s", best_model_name)
        wf_result = run_walk_forward(
            model=best_model,
            df=df,
            feature_cols=feature_cols,
            target_col=TARGET_COLUMN,
            step=walk_forward_step,
            min_train=walk_forward_min_train,
        )
        wf_metrics = wf_result["metrics"]
        wf_n = wf_result["n"]
        wf_persistence_result = run_walk_forward_persistence(
            df=df,
            feature_cols=feature_cols,
            target_col=TARGET_COLUMN,
            aqi_column="aqi",
            step=walk_forward_step,
            min_train=walk_forward_min_train,
        )
        wf_persistence_metrics = wf_persistence_result["metrics"]

    # 7) Persist the model + feature list + metadata.
    split_sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    registry_path = save_model_locally(
        model=best_model,
        model_name=best_model_name,
        feature_columns=feature_cols,
        metrics=best_metrics,
        split_sizes=split_sizes,
        package_versions=get_package_versions(),
    )

    hopsworks_registered = False
    if register_hopsworks:
        hopsworks_registered = register_model_to_hopsworks(
            run_dir=registry_path, model_name=best_model_name, metrics=best_metrics
        )

    comparison_table = build_comparison_table(val_results)
    logger.info("Training pipeline completed successfully")
    logger.info("=" * 60)

    return {
        "comparison_table": comparison_table,
        "best_model_name": best_model_name,
        "best_model_metrics": best_metrics,
        "persistence_metrics": persistence_metrics,
        "walk_forward_metrics": wf_metrics,
        "walk_forward_persistence_metrics": wf_persistence_metrics,
        "walk_forward_n": wf_n,
        "split_sizes": split_sizes,
        "registry_path": registry_path,
        "hopsworks_registered": hopsworks_registered,
    }


def main() -> int:
    """CLI entry point. Returns a process exit code (0 = success, 1 = failure)."""
    parser = argparse.ArgumentParser(description="Train and compare AQI forecasting models.")
    parser.add_argument(
        "--source",
        choices=["hopsworks", "local"],
        default="hopsworks",
        help="Where to load engineered features from (default: hopsworks).",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Fraction of rows (chronologically, earliest) used for training (default: 0.7).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of rows (chronologically, middle) used for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of rows (chronologically, most recent) used for final testing (default: 0.15).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of TimeSeriesSplit folds for hyperparameter tuning (default: {DEFAULT_N_SPLITS}).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=DEFAULT_GAP,
        help=f"Forecast-horizon gap (in samples) between CV folds (default: {DEFAULT_GAP}).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help=f"RandomizedSearchCV parameter samples per model (default: {DEFAULT_N_ITER}).",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to feature_df.csv, used only with --source local (default: data/processed/feature_df.csv).",
    )
    parser.add_argument(
        "--register-hopsworks",
        action="store_true",
        help="Also attempt to register the best model in the Hopsworks Model Registry (best-effort).",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Disable the walk-forward (expanding-window) backtest of the best model.",
    )
    parser.add_argument(
        "--walk-forward-step",
        type=int,
        default=DEFAULT_WALK_FORWARD_STEP,
        help=f"Rows per walk-forward validation block (default: {DEFAULT_WALK_FORWARD_STEP}).",
    )
    parser.add_argument(
        "--walk-forward-min-train",
        type=int,
        default=DEFAULT_WALK_FORWARD_MIN_TRAIN,
        help=f"Minimum rows in the first walk-forward training window (default: {DEFAULT_WALK_FORWARD_MIN_TRAIN}).",
    )
    args = parser.parse_args()

    try:
        results = run_training_pipeline(
            source=args.source,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.random_state,
            n_splits=args.n_splits,
            gap=args.gap,
            n_iter=args.n_iter,
            register_hopsworks=args.register_hopsworks,
            local_path=args.local_path,
            walk_forward=not args.no_walk_forward,
            walk_forward_step=args.walk_forward_step,
            walk_forward_min_train=args.walk_forward_min_train,
        )
        print("\nModel comparison (validation set, sorted by RMSE, best first):")
        print(results["comparison_table"].to_string(index=False))
        print(f"\nBest model: {results['best_model_name']}")
        print(f"Best model test metrics: {results['best_model_metrics']}")
        print(f"Persistence baseline test metrics: {results['persistence_metrics']}")
        if results["walk_forward_metrics"] is not None:
            print(f"\nWalk-forward (out-of-sample, over {results['walk_forward_n']} rows):")
            print(f"Best model walk-forward metrics: {results['walk_forward_metrics']}")
            print(
                "Persistence baseline walk-forward metrics: "
                f"{results['walk_forward_persistence_metrics']}"
            )
        print(f"Split sizes: {results['split_sizes']}")
        print(f"Saved to: {results['registry_path']}")
        if args.register_hopsworks:
            status = "succeeded" if results["hopsworks_registered"] else "failed (see log)"
            print(f"Hopsworks Model Registry push: {status}")
        return 0
    except AQIPipelineError as exc:
        logger.error("Training pipeline failed: %s", exc)
        print(f"Training pipeline failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Training pipeline failed with an unexpected error")
        print(f"Training pipeline failed with an unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

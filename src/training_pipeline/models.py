"""
Model candidates for the AQI forecasting training pipeline (Phase 5).

Provides the five tree-based model families used for the initial training
run, in evaluation order:

    1. LightGBM
    2. XGBoost
    3. CatBoost
    4. Extra Trees
    5. Random Forest

plus a **Persistence Baseline** used as a naive comparison: it predicts the
AQI ``horizon`` steps ahead using the current AQI value (the ``aqi``
feature column), which is the strongest naive benchmark for a 72-hour
forecast. Every candidate here is a tree-based ensemble; no TensorFlow,
LSTM, or GRU models are used in this phase.

Because the models are tree-based, no feature scaling or log transformation
is applied anywhere in the pipeline -- tree split-finding is invariant to
monotonic transforms and the heavily right-skewed AQI/PM2.5 distributions
are handled natively by the split thresholds.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from src.common.logger import get_logger

logger = get_logger(__name__)

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised only when deps are absent
    LGBMRegressor = None  # type: ignore[assignment,misc]
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment,misc]
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[assignment,misc]
    HAS_CATBOOST = False

# The target is AQI 72 hours (3 days) ahead. "Predict next AQI using the
# current AQI" therefore means predicting the 72h-ahead AQI from the current
# ``aqi`` feature column, which is what the persistence baseline does.
PERSISTENCE_FEATURE = "aqi"
PERSISTENCE_MODEL_NAME = "Persistence Baseline"

# Sklearn Random/Extra-Trees expose RandomState as a constructor argument;
# LightGBM/XGBoost use ``random_state``; CatBoost uses ``random_seed``.
_RANDOM_STATE_PARAM = "random_state"


class PersistenceBaselineRegressor:
    """Naive forecasting baseline: predicts the target using the current AQI.

    The target ``aqi_target`` is the AQI ``HORIZON`` hours in the future, so
    a persistence forecast simply returns the most recent observed AQI (the
    ``aqi`` feature column) for every row.

    Implements the minimal scikit-learn estimator interface
    (``fit``/``predict``) so it can be dropped into the same evaluation
    machinery as the tree-based candidates. ``fit`` is a no-op -- there are
    no learned parameters -- and ``predict`` reads the ``aqi`` column from
    ``X``.

    Attributes:
        aqi_column: Name of the feature column holding the current AQI.
    """

    def __init__(self, aqi_column: str = PERSISTENCE_FEATURE):
        self.aqi_column = aqi_column

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaselineRegressor":
        """No-op; the persistence model has no learned parameters.

        Returns:
            ``self``, for chaining (matches the scikit-learn convention).
        """
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the current AQI for each row as the persistence forecast.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``, whose
                columns follow the pipeline's ``FEATURE_COLUMNS`` order (the
                ``aqi`` column is resolved by name when ``X`` is a pandas
                DataFrame, or by the column's position otherwise).

        Returns:
            1-D array of persistence predictions, shape ``(n_samples,)``.

        Raises:
            ValueError: If ``X`` is a DataFrame without the ``aqi`` column.
        """
        if hasattr(X, "columns"):
            if self.aqi_column not in X.columns:
                raise ValueError(
                    f"Persistence baseline requires the '{self.aqi_column}' "
                    f"feature column; not found in X."
                )
            return np.asarray(X[self.aqi_column].to_numpy(), dtype="float64")
        return np.asarray(X, dtype="float64")[:, -1]  # aqi is last in FEATURE_COLUMNS


def get_model_candidates(
    random_state: int = 42, aqi_column: str = PERSISTENCE_FEATURE
) -> dict[str, Any]:
    """Build the set of tree-based model candidates to train and compare.

    Args:
        random_state: Shared random seed for reproducibility across all
            models that support one.
        aqi_column: Name of the current-AQI feature column, used by the
            persistence baseline.

    Returns:
        Dict mapping a human-readable model name to an unfitted estimator
        implementing ``fit(X, y)`` / ``predict(X)``. Candidates appear in
        the order they should be trained: LightGBM, XGBoost, CatBoost,
        Extra Trees, Random Forest, Persistence Baseline.

    Raises:
        RuntimeError: If the optional model libraries (lightgbm, xgboost,
            catboost) are not installed.
    """
    missing = [
        name
        for name, flag in (("lightgbm", HAS_LIGHTGBM), ("xgboost", HAS_XGBOOST), ("catboost", HAS_CATBOOST))
        if not flag
    ]
    if missing:
        raise RuntimeError(
            "Missing optional model dependencies: "
            f"{', '.join(missing)}. Install with: "
            "pip install lightgbm xgboost catboost"
        )

    candidates: dict[str, Any] = {
        "LightGBM": LGBMRegressor(random_state=random_state, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0),
        "CatBoost": CatBoostRegressor(
            random_seed=random_state,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        ),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
        ),
    }
    candidates[PERSISTENCE_MODEL_NAME] = PersistenceBaselineRegressor(aqi_column=aqi_column)
    return candidates


def get_param_distributions(model_name: str) -> dict[str, Any]:
    """Return the hyperparameter search space for a model candidate.

    These distributions are used with
    :class:`sklearn.model_selection.RandomizedSearchCV` (wrapped in a
    :class:`sklearn.model_selection.TimeSeriesSplit` with ``gap=72``) to
    tune each candidate before comparison.

    Args:
        model_name: A key from :func:`get_model_candidates`.

    Returns:
        A dict of ``param -> distribution`` suitable for
        ``RandomizedSearchCV.param_distributions``. The persistence
        baseline has no tunable parameters and returns ``{}``.

    Raises:
        KeyError: If ``model_name`` is not a known candidate.
    """
    if model_name == PERSISTENCE_MODEL_NAME:
        return {}

    spaces: dict[str, dict[str, Any]] = {
        "LightGBM": {
            "num_leaves": [16, 31, 64, 128],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "n_estimators": [200, 400, 600],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [5, 20, 50],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0],
        },
        "XGBoost": {
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "n_estimators": [200, 400, 600],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 10],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0],
        },
        "CatBoost": {
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "iterations": [200, 400, 600],
            "l2_leaf_reg": [1.0, 3.0, 10.0],
            "bagging_temperature": [0.0, 1.0, 5.0],
        },
        "Extra Trees": {
            "n_estimators": [200, 400, 600],
            "max_features": [0.3, 0.5, "sqrt", 1.0],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5],
            "max_depth": [None, 10, 20, 30],
        },
        "Random Forest": {
            "n_estimators": [200, 400, 600],
            "max_features": [0.3, 0.5, "sqrt", 1.0],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5],
            "max_depth": [None, 10, 20, 30],
        },
    }

    if model_name not in spaces:
        raise KeyError(f"Unknown model candidate: {model_name}")

    # Fix the seed for every candidate so tuning stays reproducible. For the
    # tree models the seed is threaded through the estimator constructor
    # below (in tune_model), so no seed is added to the search space here.
    return spaces[model_name]


def get_package_versions() -> dict[str, str]:
    """Return versions of the packages the training pipeline depends on.

    Used to record the exact environment in ``metadata.json`` so a trained
    model can be reproduced or debugged later.

    Returns:
        Dict mapping package name -> installed version string. Packages
        that are not installed are omitted.
    """
    names = ["lightgbm", "xgboost", "catboost", "scikit-learn", "numpy", "pandas", "joblib"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - env-dependent
            continue
    return versions

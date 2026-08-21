"""Model explainability (SHAP / LIME) for the daily-average AQI models.

Implements the project's explainable-AI requirement: for a given forecast,
explain *why* the model predicted that value. Because the models are Ridge
regressors trained on scaled features, the natural explainer is SHAP's
``LinearExplainer`` (exact, additive, fast). A LIME ``TabularExplainer`` is
also provided as a complementary, model-agnostic check.

Both explainers return per-feature contributions aligned with each horizon's
ordered features, which the dashboard renders as horizontal importance bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.dashboard.forecast import (
    DEFAULT_REGISTRY_DIR,
    HORIZONS,
    build_daily_features,
    load_daily_avg_models,
)

logger = get_logger(__name__)


@dataclass
class Explanation:
    """Per-feature contributions for one forecast horizon."""

    label: str
    prediction: float
    feature_cols: list[str]
    shap_values: np.ndarray
    lime_values: np.ndarray | None = None
    lime_intercept: float | None = None
    base_value: float = field(default=0.0)


def explain_forecast(
    df: pd.DataFrame,
    run_dir: Path | str | None = None,
    registry_dir: Path | str = DEFAULT_REGISTRY_DIR,
    use_lime: bool = True,
    future_weather: pd.DataFrame | None = None,
) -> list[Explanation]:
    """Generate SHAP (and optionally LIME) explanations for the latest forecast.

    Args:
        df: The hourly engineered feature dataframe (feature_df.csv).
        run_dir: Explicit registry run directory, or `None` for the latest.
        registry_dir: Root of the local model registry.
        use_lime: Also compute LIME contributions (slower). Falls back to
            SHAP-only on any LIME failure without raising.
        future_weather: Optional daily weather for the forecast days (indexed
            by date, columns `wind`, `humid`, `temp`, `clouds`, `pressure`).
            Passed through to `build_daily_features` for the v6 features.

    Returns:
        A list of one `Explanation` per horizon (day1_avg, day2_avg, day3_avg),
        aligned with `HORIZONS`.

    Raises:
        TrainingError: If models cannot be loaded or there is no scoreable row.
    """
    import shap  # imported lazily so shap is only required for this feature

    models, metadata = load_daily_avg_models(run_dir=run_dir, registry_dir=registry_dir)
    daily, _feature_cols = build_daily_features(df, future_weather=future_weather)

    all_feature_cols = sorted({c for m in metadata.values() for c in m["feature_cols"]})
    scoreable = daily.dropna(subset=all_feature_cols)
    if scoreable.empty:
        from src.common.exceptions import TrainingError

        raise TrainingError("No daily row has complete feature history to score.")

    explanations: list[Explanation] = []
    for label, _n_days in HORIZONS:
        feature_cols = metadata[label]["feature_cols"]
        model, scaler = models[label]

        X_full = scoreable[feature_cols]
        X_last = X_full.iloc[-1:]
        X_scaled_full = scaler.transform(X_full)
        X_scaled_last = scaler.transform(X_last)

        explainer = shap.LinearExplainer(model, X_scaled_full)
        shap_values = explainer.shap_values(X_scaled_last)
        base_value = float(explainer.expected_value)

        prediction = float(model.predict(X_scaled_last)[0])

        lime_values: np.ndarray | None = None
        lime_intercept: float | None = None
        if use_lime:
            try:
                lime_values, lime_intercept = _lime_explanation(
                    model,
                    scaler,
                    X_scaled_full,
                    X_scaled_last,
                    feature_cols,
                )
            except Exception as exc:  # noqa: BLE001 -- LIME is best-effort
                logger.warning("LIME explanation failed for '%s' (using SHAP only): %s", label, exc)

        explanations.append(
            Explanation(
                label=label,
                prediction=prediction,
                feature_cols=feature_cols,
                shap_values=np.asarray(shap_values).ravel(),
                lime_values=lime_values,
                lime_intercept=lime_intercept,
                base_value=base_value,
            )
        )
        logger.info(
            "Explanation for '%s': base=%.2f pred=%.2f top_feature='%s'",
            label,
            base_value,
            prediction,
            feature_cols[int(np.argmax(np.abs(np.asarray(shap_values).ravel())))],
        )

    return explanations


def _lime_explanation(
    model: Any,
    scaler: Any,
    X_scaled_full: np.ndarray,
    X_scaled_last: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, float]:
    """Fit a LIME tabular explainer and return per-feature contributions.

    Args:
        model: Fitted Ridge (or any sklearn regressor).
        scaler: Fitted StandardScaler used at train time.
        X_scaled_full: Scaled training features for LIME's background data.
        X_scaled_last: The single scaled row being explained.
        feature_cols: Ordered feature names.

    Returns:
        Tuple of `(lime_values, intercept)` where `lime_values` sums (with the
        intercept) to the model's prediction on the scaled row.
    """
    import lime.lime_tabular

    predict_fn: Callable[[np.ndarray], np.ndarray] = lambda x: model.predict(x)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_scaled_full),
        feature_names=feature_cols,
        mode="regression",
        verbose=False,
    )
    exp = explainer.explain_instance(
        np.asarray(X_scaled_last).ravel(),
        predict_fn,
        num_features=len(feature_cols),
    )

    lime_values = np.zeros(len(feature_cols))
    for feature, weight in exp.as_map()[1]:
        lime_values[feature] = weight
    return lime_values, float(exp.intercept[1])
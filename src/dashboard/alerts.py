from __future__ import annotations

AQI_BANDS = ((50, "Good", "#16a34a"), (100, "Moderate", "#ca8a04"), (150, "Unhealthy for sensitive groups", "#ea580c"), (200, "Unhealthy", "#dc2626"), (300, "Very unhealthy", "#9333ea"), (float("inf"), "Hazardous", "#7f1d1d"))


def classify_aqi(value: float) -> tuple[str, str]:
    for upper, label, color in AQI_BANDS:
        if value <= upper:
            return label, color
    return AQI_BANDS[-1][1], AQI_BANDS[-1][2]


def trend_delta(value: float, baseline: float, stable_threshold: float = 2.0) -> tuple[str, float]:
    delta = value - baseline
    if abs(delta) < stable_threshold:
        return "Stable", delta
    return ("Increasing" if delta > 0 else "Decreasing"), delta


def alert_level(value: float) -> str | None:
    label, _ = classify_aqi(value)
    return label if value >= 150 else None

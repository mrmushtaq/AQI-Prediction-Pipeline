$chunks = @(
    @("2024-01-01","2024-03-31"),
    @("2024-04-01","2024-06-30"),
    @("2024-07-01","2024-09-30"),
    @("2024-10-01","2024-12-31"),
    @("2025-01-01","2025-03-31"),
    @("2025-04-01","2025-06-30"),
    @("2025-07-01","2025-09-30"),
    @("2025-10-01","2025-12-31"),
    @("2026-01-01","2026-03-31"),
    @("2026-04-01","2026-06-30"),
    @("2026-07-01","2026-07-30")
)

foreach ($chunk in $chunks) {
    $start = $chunk[0]
    $end = $chunk[1]
    Write-Host "=========================================="
    Write-Host "Running backfill: $start to $end"
    Write-Host "=========================================="
    python -m src.feature_pipeline.historical_backfill $start $end
    $rowCount = (python -c "import pandas as pd; print(len(pd.read_csv('data/processed/aqi_dataset_historical.csv')))")
    Write-Host "Chunk done. Total rows so far: $rowCount"
}

Write-Host "ALL CHUNKS COMPLETE"
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from .fetcher import build_aligned_dataset
from .analytics import apply_rolling_bias_correction, calculate_metrics, get_daily_breakdown

app = FastAPI(title="UK Wind Power Forecast Analytics")

# Enable CORS for local testing and cloud deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_state = {}


@app.on_event("startup")
def load_data():
    raw_df = build_aligned_dataset()
    dataset_state["raw"] = raw_df
    dataset_state["processed"] = apply_rolling_bias_correction(raw_df, target_quantile=0.60)
    print("Application startup complete. Baseline quantile q=0.60 calibrated.")


@app.get("/api/health")
def health():
    return {"status": "ok", "records": len(dataset_state.get("raw", []))}


@app.get("/api/data")
def get_data(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    quantile: float = Query(0.60, ge=0.10, le=0.90),
):
    df = dataset_state.get("raw")
    if df is None or df.empty:
        return []

    # Recalibrate dynamically if custom quantile is selected
    if abs(quantile - 0.60) > 0.001:
        processed = apply_rolling_bias_correction(df, target_quantile=quantile)
    else:
        processed = dataset_state["processed"]

    filtered = processed.copy()
    if start_date:
        filtered = filtered[filtered["startTime"] >= f"{start_date}T00:00:00Z"]
    if end_date:
        filtered = filtered[filtered["startTime"] <= f"{end_date}T23:59:59Z"]

    return filtered.to_dict(orient="records")


@app.get("/api/metrics")
def get_metrics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    quantile: float = Query(0.60, ge=0.10, le=0.90),
):
    df = dataset_state.get("raw")
    if df is None or df.empty:
        return {}

    if abs(quantile - 0.60) > 0.001:
        processed = apply_rolling_bias_correction(df, target_quantile=quantile)
    else:
        processed = dataset_state["processed"]

    filtered = processed.copy()
    if start_date:
        filtered = filtered[filtered["startTime"] >= f"{start_date}T00:00:00Z"]
    if end_date:
        filtered = filtered[filtered["startTime"] <= f"{end_date}T23:59:59Z"]

    return calculate_metrics(filtered, target_quantile=quantile)


@app.get("/api/metrics/daily")
def get_daily_metrics():
    df = dataset_state.get("processed")
    if df is None or df.empty:
        return []
    return get_daily_breakdown(df)

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd

from .fetcher import build_aligned_dataset
from .analytics import calculate_metrics, get_daily_breakdown

app = FastAPI(
    title="UK Wind Power Forecast Analytics API",
    description="Day-ahead forecast error evaluation for wholesale energy trading",
    version="1.0.0"
)

# Enable CORS for React frontend (local dev and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dataframe cache loaded at startup
df_cache: pd.DataFrame = pd.DataFrame()


from .analytics import calculate_metrics, get_daily_breakdown, apply_rolling_bias_correction

@app.on_event("startup")
def load_data():
    global df_cache
    print("Loading aligned January 2026 wind power data...")
    raw_df = build_aligned_dataset()
    df_cache = apply_rolling_bias_correction(raw_df)
    print(f"Data ready with {len(df_cache)} records and bias correction applied.")



@app.get("/api/health")
def health_check():
    return {"status": "ok", "records_loaded": len(df_cache)}


@app.get("/api/data")
def get_time_series_data(
    start_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD")
):
    """Returns 30-minute settlement period series filtered by optional date range."""
    global df_cache
    if df_cache.empty:
        return []

    filtered = df_cache.copy()
    if start_date:
        filtered = filtered[filtered['startTime'] >= f"{start_date}T00:00:00Z"]
    if end_date:
        filtered = filtered[filtered['startTime'] <= f"{end_date}T23:59:59Z"]

    return filtered.to_dict(orient="records")


@app.get("/api/metrics")
def get_metrics_summary(
    start_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD")
):
    """Calculates summary KPIs (MAE, RMSE, Bias, MAPE, totals) for the selected range."""
    global df_cache
    filtered = df_cache.copy()
    if start_date:
        filtered = filtered[filtered['startTime'] >= f"{start_date}T00:00:00Z"]
    if end_date:
        filtered = filtered[filtered['startTime'] <= f"{end_date}T23:59:59Z"]

    return calculate_metrics(filtered)


@app.get("/api/metrics/daily")
def get_daily_metrics():
    """Returns daily trend of MAE, RMSE, and bias across the month."""
    global df_cache
    return get_daily_breakdown(df_cache)

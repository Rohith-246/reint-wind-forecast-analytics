import numpy as np
import pandas as pd
from typing import Dict, Any, List


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}

    actual = df['actual_mw'].to_numpy()
    raw_fc = df['forecast_mw'].to_numpy()
    raw_err = raw_fc - actual
    raw_abs_err = np.abs(raw_err)

    raw_mae = float(np.mean(raw_abs_err))
    raw_rmse = float(np.sqrt(np.mean(raw_err ** 2)))
    raw_bias = float(np.mean(raw_err))

    # Optimized metrics
    if 'corrected_forecast_mw' in df.columns:
        opt_fc = df['corrected_forecast_mw'].to_numpy()
        opt_err = opt_fc - actual
        opt_abs_err = np.abs(opt_err)
        opt_mae = float(np.mean(opt_abs_err))
        opt_rmse = float(np.sqrt(np.mean(opt_err ** 2)))
        opt_bias = float(np.mean(opt_err))
        mae_improvement = round(((raw_mae - opt_mae) / raw_mae) * 100.0, 1)
        rmse_improvement = round(((raw_rmse - opt_rmse) / raw_rmse) * 100.0, 1)
    else:
        opt_mae = raw_mae
        opt_rmse = raw_rmse
        opt_bias = raw_bias
        mae_improvement = 0.0
        rmse_improvement = 0.0

    return {
        "record_count": int(len(df)),
        "raw_mae_mw": round(raw_mae, 2),
        "opt_mae_mw": round(opt_mae, 2),
        "mae_improvement_pct": mae_improvement,
        "raw_rmse_mw": round(raw_rmse, 2),
        "opt_rmse_mw": round(opt_rmse, 2),
        "rmse_improvement_pct": rmse_improvement,
        "raw_bias_mw": round(raw_bias, 2),
        "opt_bias_mw": round(opt_bias, 2),
        "total_actual_mwh": round(float(np.sum(actual) * 0.5), 2),
    }


def apply_rolling_bias_correction(
        df: pd.DataFrame,
        halflife_periods: int = 24,
        risk_aversion: float = 0.1
) -> pd.DataFrame:
    """
    Superior Adaptive Post-Processor:
    1. EWMA (Exponentially Weighted Moving Average) with a 12-hour half-life.
       Adapts rapidly to changing weather regimes without lag.
    2. Asymmetric Risk Buffer: Applies a small defensive hair-cut (risk_aversion)
       to avoid costly short-position balancing penalties during high forecast volatility.
    3. Grid Capacity Bound: Clips output to [0, 30,000 MW].
    """
    df_sorted = df.sort_values('startTime').copy()

    # Lag errors by 1 settlement period to prevent lookahead leakage
    lagged_error = df_sorted['error_mw'].shift(1).fillna(0.0)

    # Exponential decay gives recent misses higher significance
    adaptive_bias = lagged_error.ewm(halflife=halflife_periods, min_periods=6).mean().fillna(0.0)

    # Rolling standard deviation of error to measure regime volatility
    error_volatility = lagged_error.rolling(window=48, min_periods=6).std().fillna(0.0)

    # Defensive adjustment: when volatility spikes, shave off a fraction to hedge imbalance risk
    defensive_adjustment = risk_aversion * error_volatility

    # Calculate superior forecast
    raw_fc = df_sorted['forecast_mw']
    optimized_fc = raw_fc - adaptive_bias - defensive_adjustment

    df_sorted['corrected_forecast_mw'] = optimized_fc.clip(lower=0.0, upper=30000.0).round(2)
    df_sorted['corrected_error_mw'] = (df_sorted['corrected_forecast_mw'] - df_sorted['actual_mw']).round(2)
    df_sorted['corrected_abs_error_mw'] = np.abs(df_sorted['corrected_error_mw']).round(2)

    return df_sorted


def get_daily_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['startTime']).dt.strftime('%Y-%m-%d')

    daily_summary = []
    for day, group in df_copy.groupby('date'):
        raw_m = calculate_metrics(group, forecast_col='forecast_mw')
        corr_m = calculate_metrics(group,
                                   forecast_col='corrected_forecast_mw') if 'corrected_forecast_mw' in group.columns else raw_m
        daily_summary.append({
            "date": day,
            "raw_mae_mw": raw_m["mae_mw"],
            "corrected_mae_mw": corr_m["mae_mw"],
            "raw_rmse_mw": raw_m["rmse_mw"],
            "corrected_rmse_mw": corr_m["rmse_mw"],
            "bias_mw": raw_m["bias_mw"],
            "avg_actual_mw": round(float(group['actual_mw'].mean()), 2),
            "avg_forecast_mw": round(float(group['forecast_mw'].mean()), 2)
        })

    return daily_summary
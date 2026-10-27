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


from typing import Any, Dict
import numpy as np
import pandas as pd


def apply_rolling_bias_correction(
    df: pd.DataFrame,
    halflife_periods: int = 18,  # 9-hour half-life for fast weather front adaptation
    volatility_window: int = 48, # 24-hour rolling regime tracking
    capacity_mw: float = 30000.0,
    asymmetry_factor: float = 0.08
) -> pd.DataFrame:
    """
    High-Efficiency Regime-Aware Adaptive Forecast Corrector:
    1. Fast EWMA bias correction with strictly lagged feedback (no data leakage).
    2. Dynamic error volatility damping (asymmetric cash-out risk protection).
    3. Sigmoid/Power-curve saturation scaling near grid boundary limits [0, capacity_mw].
    """
    df_sorted = df.sort_values('startTime').copy()

    # Calculate raw forecast error lagged by 1 period to prevent lookahead
    # Error = Forecast - Actual (positive means raw model over-forecasted)
    raw_error = df_sorted['forecast_mw'] - df_sorted['actual_mw']
    lagged_error = raw_error.shift(1).fillna(0.0)

    # 1. Fast adaptive EWMA bias tracker
    adaptive_bias = (
        lagged_error.ewm(halflife=halflife_periods, min_periods=4)
        .mean()
        .fillna(0.0)
    )

    # 2. Regime volatility tracking (detects erratic weather ramps)
    error_volatility = (
        lagged_error.rolling(window=volatility_window, min_periods=4)
        .std()
        .fillna(0.0)
    )

    # 3. Dynamic asymmetric hedge (protects against expensive system short positions)
    downside_hedge = asymmetry_factor * error_volatility

    # 4. Power-curve boundary dampening:
    # Scale adjustment down as forecast approaches 0 MW or max grid capacity
    normalized_pos = (df_sorted['forecast_mw'] / capacity_mw).clip(0.0, 1.0)
    # Quadratic dampener that drops to 0 at edges and peaks in the linear middle band
    saturation_weight = 4.0 * normalized_pos * (1.0 - normalized_pos)

    effective_adjustment = (adaptive_bias + downside_hedge) * saturation_weight

    # Optimized generation estimate
    optimized = df_sorted['forecast_mw'] - effective_adjustment

    # Physical clamping to UK wind capacity envelope
    df_sorted['corrected_forecast_mw'] = optimized.clip(lower=0.0, upper=capacity_mw).round(2)
    df_sorted['corrected_error_mw'] = (df_sorted['corrected_forecast_mw'] - df_sorted['actual_mw']).round(2)
    df_sorted['corrected_abs_error_mw'] = np.abs(df_sorted['corrected_error_mw']).round(2)

    return df_sorted


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
        opt_mae, opt_rmse, opt_bias = raw_mae, raw_rmse, raw_bias
        mae_improvement, rmse_improvement = 0.0, 0.0

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
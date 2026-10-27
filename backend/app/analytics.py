"""
Advanced Quantitative Wind Power Analytics & Multi-Horizon Bias Correction Engine.
Features:
  - Multi-quantile LightGBM regression (P10, P50/Target, P90 confidence envelopes)
  - UK peak-hour Balancing Mechanism market cash-out sample weighting
  - Aerodynamic power-curve slope sensitivity index
  - Physical fleet ramp-rate constraints (dP/dt <= 1600 MW / 30 min)
  - Daily breakdown reporting for API consumer endpoints
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import lightgbm as lgb


def enforce_fleet_ramp_limits(
    forecast_series: pd.Series,
    actual_series: pd.Series,
    max_ramp_mw_per_period: float = 1600.0,
) -> pd.Series:
    """Enforces aerodynamic turbine fleet ramp limits (dP/dt)."""
    values = forecast_series.to_numpy(copy=True)
    actuals = actual_series.to_numpy()

    for i in range(1, len(values)):
        prior_ref = actuals[i - 1] if not np.isnan(actuals[i - 1]) else values[i - 1]
        delta = values[i] - prior_ref

        if delta > max_ramp_mw_per_period:
            values[i] = prior_ref + max_ramp_mw_per_period
        elif delta < -max_ramp_mw_per_period:
            values[i] = prior_ref - max_ramp_mw_per_period

    return pd.Series(values, index=forecast_series.index)


def train_and_predict_quantiles(
    df: pd.DataFrame, target_quantile: float = 0.60
) -> Dict[str, np.ndarray]:
    """Trains multi-quantile LightGBM regressors (P10, Target, P90) with peak cash-out weights."""
    df_feat = df.copy()

    # 1. Temporal & Cyclical Harmonic Encodings
    times = pd.to_datetime(df_feat["startTime"])
    sp = df_feat["settlementPeriod"]
    df_feat["sp_sin"] = np.sin(2.0 * np.pi * sp / 48.0)
    df_feat["sp_cos"] = np.cos(2.0 * np.pi * sp / 48.0)
    df_feat["dow"] = times.dt.dayofweek

    # 2. Aerodynamic Power Curve Sensitivity & Physical Ramps
    df_feat["fc_ramp"] = df_feat["forecast_mw"].diff().fillna(0.0)
    df_feat["fc_accel"] = df_feat["fc_ramp"].diff().fillna(0.0)

    # Non-linear turbine sensitivity (highest in Region 2 cubic growth around 50% capacity)
    capacity_mw = 30000.0
    norm_pos = (df_feat["forecast_mw"] / capacity_mw).clip(0.0, 1.0)
    df_feat["aero_sensitivity"] = np.sin(np.pi * norm_pos)

    # 3. Strictly Lagged Error Features (No Lookahead Leakage)
    raw_error = df_feat["forecast_mw"] - df_feat["actual_mw"]
    df_feat["lag_err_1"] = raw_error.shift(1).fillna(0.0)
    df_feat["lag_err_2"] = raw_error.shift(2).fillna(0.0)
    df_feat["lag_err_48"] = raw_error.shift(48).fillna(0.0)
    df_feat["err_std_12"] = (
        df_feat["lag_err_1"].rolling(12, min_periods=2).std().fillna(0.0)
    )

    feature_cols = [
        "sp_sin",
        "sp_cos",
        "dow",
        "fc_ramp",
        "fc_accel",
        "aero_sensitivity",
        "lag_err_1",
        "lag_err_2",
        "lag_err_48",
        "err_std_12",
        "forecast_mw",
    ]

    X = df_feat[feature_cols]
    y = raw_error

    # Walk-forward calibration size
    train_size = max(48, int(len(df_feat) * 0.35))
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]

    # UK Peak Hour Loss Weighting (Settlement Periods 33-38: Evening Peak 16:30 - 19:30)
    sp_train = sp.iloc[:train_size]
    sample_weights = np.where(sp_train.between(33, 38), 2.2, 1.0)

    # Multi-quantile predictions
    quantiles = {
        "p10": 0.10,
        "target": float(target_quantile),
        "p90": 0.90,
    }
    predictions = {}

    for name, q in quantiles.items():
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=50,
            max_depth=4,
            learning_rate=0.06,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        predictions[name] = model.predict(X)

    return predictions


def apply_rolling_bias_correction(
    df: pd.DataFrame,
    target_quantile: float = 0.60,
    halflife_periods: int = 18,
    capacity_mw: float = 30000.0,
) -> pd.DataFrame:
    """Full quantitative hybrid optimization pipeline generating point and confidence bounds."""
    df_sorted = df.sort_values("startTime").copy()

    if "settlementPeriod" not in df_sorted.columns:
        times = pd.to_datetime(df_sorted["startTime"])
        df_sorted["settlementPeriod"] = (
            times.dt.hour * 2 + times.dt.minute // 30 + 1
        )

    # 1. Cyclical Diurnal Bias
    raw_error = df_sorted["forecast_mw"] - df_sorted["actual_mw"]
    lagged_error = raw_error.shift(1).fillna(0.0)
    df_sorted["lagged_error"] = lagged_error

    diurnal_bias = (
        df_sorted.groupby("settlementPeriod")["lagged_error"]
        .transform(lambda s: s.shift(1).expanding(min_periods=2).median())
        .fillna(0.0)
    )

    # 2. Adaptive EWMA Residual
    adaptive_residual = (
        (lagged_error - diurnal_bias)
        .ewm(halflife=halflife_periods, min_periods=4)
        .mean()
        .fillna(0.0)
    )
    heuristic_bias = diurnal_bias + adaptive_residual

    # 3. Train Multi-Quantile Residual Models
    q_preds = train_and_predict_quantiles(df_sorted, target_quantile=target_quantile)

    # 4. Ensembling & Boundary Clamping
    for q_name, res_pred in q_preds.items():
        weight_ml = 0.60 if q_name == "target" else 0.85
        blended = (1.0 - weight_ml) * heuristic_bias + weight_ml * res_pred
        raw_output = (df_sorted["forecast_mw"] - blended).clip(lower=0.0, upper=capacity_mw)

        if q_name == "target":
            df_sorted["corrected_forecast_mw"] = enforce_fleet_ramp_limits(
                raw_output, df_sorted["actual_mw"]
            ).round(2)
        elif q_name == "p10":
            df_sorted["forecast_p10_mw"] = raw_output.round(2)
        elif q_name == "p90":
            df_sorted["forecast_p90_mw"] = raw_output.round(2)

    # 5. Error Calculations
    df_sorted["error_mw"] = (
        df_sorted["forecast_mw"] - df_sorted["actual_mw"]
    ).round(2)
    df_sorted["corrected_error_mw"] = (
        df_sorted["corrected_forecast_mw"] - df_sorted["actual_mw"]
    ).round(2)
    df_sorted["corrected_abs_error_mw"] = np.abs(
        df_sorted["corrected_error_mw"]
    ).round(2)

    return df_sorted


def calculate_metrics(df: pd.DataFrame, target_quantile: float = 0.60) -> Dict[str, Any]:
    """Calculates summary accuracy, volume, and cash-out pinball loss metrics."""
    if df.empty:
        return {}

    actual = df["actual_mw"].to_numpy()
    raw_fc = df["forecast_mw"].to_numpy()
    raw_err = raw_fc - actual
    raw_abs_err = np.abs(raw_err)

    raw_mae = float(np.mean(raw_abs_err))
    raw_rmse = float(np.sqrt(np.mean(raw_err**2)))
    raw_bias = float(np.mean(raw_err))

    q = float(target_quantile)
    raw_pinball = float(np.mean(np.maximum(q * raw_err, (q - 1) * raw_err)))

    if "corrected_forecast_mw" in df.columns:
        opt_fc = df["corrected_forecast_mw"].to_numpy()
        opt_err = opt_fc - actual
        opt_abs_err = np.abs(opt_err)

        opt_mae = float(np.mean(opt_abs_err))
        opt_rmse = float(np.sqrt(np.mean(opt_err**2)))
        opt_bias = float(np.mean(opt_err))
        opt_pinball = float(np.mean(np.maximum(q * opt_err, (q - 1) * opt_err)))

        mae_improvement = round(((raw_mae - opt_mae) / raw_mae) * 100.0, 1)
        rmse_improvement = round(((raw_rmse - opt_rmse) / raw_rmse) * 100.0, 1)
        risk_reduction = round(
            ((raw_pinball - opt_pinball) / raw_pinball) * 100.0, 1
        )
    else:
        opt_mae, opt_rmse, opt_bias, opt_pinball = raw_mae, raw_rmse, raw_bias, raw_pinball
        mae_improvement, rmse_improvement, risk_reduction = 0.0, 0.0, 0.0

    return {
        "record_count": int(len(df)),
        "target_quantile": q,
        "raw_mae_mw": round(raw_mae, 2),
        "opt_mae_mw": round(opt_mae, 2),
        "mae_improvement_pct": mae_improvement,
        "raw_rmse_mw": round(raw_rmse, 2),
        "opt_rmse_mw": round(opt_rmse, 2),
        "rmse_improvement_pct": rmse_improvement,
        "raw_bias_mw": round(raw_bias, 2),
        "opt_bias_mw": round(opt_bias, 2),
        "raw_imbalance_penalty_mw": round(raw_pinball, 2),
        "opt_imbalance_penalty_mw": round(opt_pinball, 2),
        "imbalance_risk_reduction_pct": risk_reduction,
        "total_actual_mwh": round(float(np.sum(actual) * 0.5), 2),
    }


def get_daily_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generates day-by-day aggregations."""
    if df.empty:
        return []

    df_copy = df.copy()
    times = pd.to_datetime(df_copy["startTime"])
    df_copy["date"] = times.dt.strftime("%Y-%m-%d")

    daily_rows = []
    for date_str, group in df_copy.groupby("date"):
        actual = group["actual_mw"].to_numpy()
        raw_fc = group["forecast_mw"].to_numpy()
        raw_err = raw_fc - actual
        raw_mae = float(np.mean(np.abs(raw_err)))
        raw_rmse = float(np.sqrt(np.mean(raw_err**2)))

        if "corrected_forecast_mw" in group.columns:
            opt_fc = group["corrected_forecast_mw"].to_numpy()
            opt_err = opt_fc - actual
            opt_mae = float(np.mean(np.abs(opt_err)))
            opt_rmse = float(np.sqrt(np.mean(opt_err**2)))
            mae_improvement = round(((raw_mae - opt_mae) / raw_mae) * 100.0, 1)
        else:
            opt_mae, opt_rmse, mae_improvement = raw_mae, raw_rmse, 0.0

        daily_rows.append({
            "date": date_str,
            "raw_mae_mw": round(raw_mae, 2),
            "opt_mae_mw": round(opt_mae, 2),
            "raw_rmse_mw": round(raw_rmse, 2),
            "opt_rmse_mw": round(opt_rmse, 2),
            "mae_improvement_pct": mae_improvement,
            "total_mwh": round(float(np.sum(actual) * 0.5), 2),
        })

    return daily_rows

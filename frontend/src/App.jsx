import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const API_BASE = (import.meta.env.VITE_API_URL || 'https://reint-wind-forecast-analytics.onrender.com') + '/api';

export default function App() {
  const [data, setData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [daily, setDaily] = useState([]);
  const [loading, setLoading] = useState(true);
  const [startDate, setStartDate] = useState('2026-01-01');
  const [endDate, setEndDate] = useState('2026-01-07');
  const [quantile, setQuantile] = useState(0.60);
  const [showBands, setShowBands] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [resData, resMetrics, resDaily] = await Promise.all([
        fetch(`${API_BASE}/data?start_date=${startDate}&end_date=${endDate}&quantile=${quantile}`),
        fetch(`${API_BASE}/metrics?start_date=${startDate}&end_date=${endDate}&quantile=${quantile}`),
        fetch(`${API_BASE}/metrics/daily`)
      ]);

      const [jsonD, jsonM, jsonDaily] = await Promise.all([
        resData.json(),
        resMetrics.json(),
        resDaily.json()
      ]);

      setData(Array.isArray(jsonD) ? jsonD : []);
      setMetrics(jsonM);
      setDaily(Array.isArray(jsonDaily) ? jsonDaily : []);
    } catch (err) {
      console.error("Failed to load analytics data", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [startDate, endDate, quantile]);

  return (
    <div style={{ padding: '24px', backgroundColor: '#f8fafc', minHeight: '100vh', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color: '#0f172a' }}>
      
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '26px', fontWeight: '800', letterSpacing: '-0.02em', color: '#0f172a' }}>
            UK Wind Power Day-Ahead Forecasting & Cash-Out Analytics
          </h1>
          <p style={{ margin: '4px 0 0 0', color: '#64748b', fontSize: '14px' }}>
            Elexon BMRS Stream Processing (January 2026) · LightGBM Quantile Optimization
          </p>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <span style={{ backgroundColor: '#e0f2fe', color: '#0369a1', padding: '6px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: '600' }}>
            Elexon BMRS Live
          </span>
          <span style={{ backgroundColor: '#dcfce7', color: '#15803d', padding: '6px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: '600' }}>
            Multi-Quantile (P10/P90) Active
          </span>
        </div>
      </div>

      {/* Control Bar with Slider */}
      <div style={{ backgroundColor: '#ffffff', padding: '16px 20px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        
        {/* Date Filter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '13px', fontWeight: '600', color: '#475569' }}>Date Range:</span>
          <input
            type="date"
            value={startDate}
            min="2026-01-01"
            max="2026-01-31"
            onChange={(e) => setStartDate(e.target.value)}
            style={{ padding: '6px 10px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '13px' }}
          />
          <span style={{ color: '#94a3b8' }}>to</span>
          <input
            type="date"
            value={endDate}
            min="2026-01-01"
            max="2026-01-31"
            onChange={(e) => setEndDate(e.target.value)}
            style={{ padding: '6px 10px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '13px' }}
          />
        </div>

        {/* Dynamic Trader Risk Slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', backgroundColor: '#f1f5f9', padding: '8px 16px', borderRadius: '8px' }}>
          <span style={{ fontSize: '13px', fontWeight: '700', color: '#1e293b' }}>
            Risk Posture (α = {Number(quantile).toFixed(2)}):
          </span>
          <input
            type="range"
            min="0.50"
            max="0.80"
            step="0.05"
            value={quantile}
            onChange={(e) => setQuantile(parseFloat(e.target.value))}
            style={{ width: '130px', cursor: 'pointer' }}
          />
          <button
            onClick={() => setShowBands(!showBands)}
            style={{ padding: '4px 10px', borderRadius: '6px', border: '1px solid #cbd5e1', backgroundColor: '#ffffff', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}
          >
            {showBands ? "Hide P10/P90" : "Show P10/P90"}
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      {metrics && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '24px' }}>
          <div style={{ backgroundColor: '#ffffff', padding: '18px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ color: '#64748b', fontSize: '13px', fontWeight: '600' }}>MAE (Mean Absolute Error)</div>
            <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '8px', color: '#0f172a' }}>{metrics.opt_mae_mw} MW</div>
            <div style={{ fontSize: '12px', color: '#16a34a', fontWeight: '600', marginTop: '4px' }}>
              ↓ {metrics.mae_improvement_pct}% vs Raw ({metrics.raw_mae_mw} MW)
            </div>
          </div>

          <div style={{ backgroundColor: '#ffffff', padding: '18px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ color: '#64748b', fontSize: '13px', fontWeight: '600' }}>RMSE (Tail Volatility)</div>
            <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '8px', color: '#0f172a' }}>{metrics.opt_rmse_mw} MW</div>
            <div style={{ fontSize: '12px', color: '#16a34a', fontWeight: '600', marginTop: '4px' }}>
              ↓ {metrics.rmse_improvement_pct}% vs Raw ({metrics.raw_rmse_mw} MW)
            </div>
          </div>

          <div style={{ backgroundColor: '#ffffff', padding: '18px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ color: '#64748b', fontSize: '13px', fontWeight: '600' }}>Cash-Out Imbalance Penalty</div>
            <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '8px', color: '#0f172a' }}>{metrics.opt_imbalance_penalty_mw} MW</div>
            <div style={{ fontSize: '12px', color: '#16a34a', fontWeight: '600', marginTop: '4px' }}>
              ↓ {metrics.imbalance_risk_reduction_pct}% Risk Reduction
            </div>
          </div>

          <div style={{ backgroundColor: '#ffffff', padding: '18px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ color: '#64748b', fontSize: '13px', fontWeight: '600' }}>Total Delivered Volume</div>
            <div style={{ fontSize: '24px', fontWeight: '800', marginTop: '8px', color: '#0f172a' }}>{metrics.total_actual_mwh.toLocaleString()} MWh</div>
            <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>Across {metrics.record_count} Settlement Periods</div>
          </div>
        </div>
      )}

      {/* Main Graph */}
      <div style={{ backgroundColor: '#ffffff', padding: '24px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', marginBottom: '24px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '700', color: '#1e293b' }}>
          Wind Generation vs Day-Ahead Forecast with P10/P90 Bounds (MW)
        </h3>
        <div style={{ height: '380px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="startTime" tickFormatter={(v) => v.slice(5, 16).replace('T', ' ')} stroke="#94a3b8" fontSize={11} />
              <YAxis unit=" MW" stroke="#94a3b8" fontSize={11} />
              <Tooltip labelFormatter={(l) => `Time: ${l.replace('T', ' ').replace('Z', '')}`} />
              <Legend />
              
              {showBands && (
                <>
                  <Line type="monotone" dataKey="forecast_p90_mw" name="Upper P90 Bound" stroke="#cbd5e1" strokeDasharray="4 4" dot={false} strokeWidth={1} />
                  <Line type="monotone" dataKey="forecast_p10_mw" name="Lower P10 Bound" stroke="#cbd5e1" strokeDasharray="4 4" dot={false} strokeWidth={1} />
                </>
              )}
              
              <Line type="monotone" dataKey="actual_mw" name="Actual Wind (FUELHH)" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="forecast_mw" name="Raw Day-Ahead (WINDFOR)" stroke="#94a3b8" strokeDasharray="3 3" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="corrected_forecast_mw" name={`Optimized Model (q=${quantile})`} stroke="#6366f1" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Residuals Error Chart */}
      <div style={{ backgroundColor: '#ffffff', padding: '24px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', marginBottom: '24px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '700', color: '#1e293b' }}>
          Residual Error Variance Comparison (MW)
        </h3>
        <div style={{ height: '240px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="startTime" tickFormatter={(v) => v.slice(5, 16).replace('T', ' ')} stroke="#94a3b8" fontSize={11} />
              <YAxis unit=" MW" stroke="#94a3b8" fontSize={11} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="error_mw" name="Raw Forecast Error (MW)" stroke="#f87171" strokeDasharray="2 2" strokeWidth={1} dot={false} />
              <Line type="monotone" dataKey="corrected_error_mw" name="Optimized Model Residual (MW)" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Daily Breakdown Table */}
      <div style={{ backgroundColor: '#ffffff', padding: '24px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '700', color: '#1e293b' }}>
          Daily Trajectory Breakdown
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ padding: '10px 12px' }}>Date</th>
                <th style={{ padding: '10px 12px' }}>Actual Volume</th>
                <th style={{ padding: '10px 12px' }}>Raw MAE</th>
                <th style={{ padding: '10px 12px' }}>Optimized MAE</th>
                <th style={{ padding: '10px 12px' }}>MAE Improvement</th>
                <th style={{ padding: '10px 12px' }}>Raw RMSE</th>
                <th style={{ padding: '10px 12px' }}>Optimized RMSE</th>
              </tr>
            </thead>
            <tbody>
              {daily.map((row) => (
                <tr key={row.date} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '10px 12px', fontWeight: '600' }}>{row.date}</td>
                  <td style={{ padding: '10px 12px' }}>{row.total_mwh.toLocaleString()} MWh</td>
                  <td style={{ padding: '10px 12px', color: '#ef4444' }}>{row.raw_mae_mw} MW</td>
                  <td style={{ padding: '10px 12px', color: '#16a34a', fontWeight: '600' }}>{row.opt_mae_mw} MW</td>
                  <td style={{ padding: '10px 12px', color: '#16a34a', fontWeight: '600' }}>+{row.mae_improvement_pct}%</td>
                  <td style={{ padding: '10px 12px' }}>{row.raw_rmse_mw} MW</td>
                  <td style={{ padding: '10px 12px' }}>{row.opt_rmse_mw} MW</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}

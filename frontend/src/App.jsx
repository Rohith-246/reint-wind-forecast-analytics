import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000/api";

export default function App() {
  const [metrics, setMetrics] = useState(null);
  const [seriesData, setSeriesData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [startDate, setStartDate] = useState("2026-01-01");
  const [endDate, setEndDate] = useState("2026-01-07");

  useEffect(() => {
    fetchData();
  }, [startDate, endDate]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const q = `?start_date=${startDate}&end_date=${endDate}`;
      const [resMetrics, resSeries] = await Promise.all([
        fetch(`${API_BASE}/metrics${q}`).then((r) => r.json()),
        fetch(`${API_BASE}/data${q}`).then((r) => r.json()),
      ]);

      setMetrics(resMetrics);
      const formattedSeries = resSeries.map((item) => ({
        ...item,
        timeLabel: item.startTime.substring(5, 16).replace("T", " "),
      }));
      setSeriesData(formattedSeries);
    } catch (err) {
      console.error("Failed to fetch analytics data", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1>UK Wind Power Forecast & Error Analytics</h1>
          <p>Day-Ahead wholesale procurement decision support & ML post-processing (January 2026)</p>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          <span className="badge">Elexon BMRS Live</span>
          <span className="badge" style={{ backgroundColor: "#dcfce7", color: "#15803d" }}>
            Adaptive EWMA Bias Correction Active
          </span>
        </div>
      </header>

      {/* Date Filter Controls */}
      <div className="controls">
        <label>
          <strong>From:</strong>{" "}
          <input
            type="date"
            min="2026-01-01"
            max="2026-01-31"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </label>
        <label>
          <strong>To:</strong>{" "}
          <input
            type="date"
            min="2026-01-01"
            max="2026-01-31"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </label>
      </div>

      {/* KPI Cards */}
      {metrics && (
        <div className="kpi-grid">
          <div className="kpi-card">
            <div className="label">MAE (Raw vs Optimized)</div>
            <div className="value">{metrics.opt_mae_mw} MW</div>
            <div className="subtext" style={{ color: "#16a34a", fontWeight: "600" }}>
              ↓ {metrics.mae_improvement_pct}% reduction (from {metrics.raw_mae_mw} MW)
            </div>
          </div>
          <div className="kpi-card">
            <div className="label">RMSE Outlier Penalty</div>
            <div className="value">{metrics.opt_rmse_mw} MW</div>
            <div className="subtext" style={{ color: "#16a34a", fontWeight: "600" }}>
              ↓ {metrics.rmse_improvement_pct}% reduction (from {metrics.raw_rmse_mw} MW)
            </div>
          </div>
          <div className="kpi-card">
            <div className="label">Systemic Bias Corrected</div>
            <div className="value" style={{ color: "#2563eb" }}>
              {metrics.opt_bias_mw > 0 ? `+${metrics.opt_bias_mw}` : metrics.opt_bias_mw} MW
            </div>
            <div className="subtext">
              Reduced from {metrics.raw_bias_mw > 0 ? `+${metrics.raw_bias_mw}` : metrics.raw_bias_mw} MW raw bias
            </div>
          </div>
          <div className="kpi-card">
            <div className="label">Total Wind Volume</div>
            <div className="value">{(metrics.total_actual_mwh / 1000).toFixed(1)} GWh</div>
            <div className="subtext">January 2026 total delivered energy</div>
          </div>
        </div>
      )}

      {/* Actual vs Raw Forecast vs Corrected Forecast */}
      <div className="chart-card">
        <h2>Wind Generation vs Day-Ahead Forecast & Optimized Model (MW)</h2>
        {loading ? (
          <p>Updating curves...</p>
        ) : (
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={seriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="timeLabel" minTickGap={40} tick={{ fontSize: 12 }} />
              <YAxis
                tick={{ fontSize: 12 }}
                label={{ value: "Power (MW)", angle: -90, position: "insideLeft" }}
              />
              <Tooltip />
              <Legend verticalAlign="top" height={36} />
              <Line
                type="monotone"
                dataKey="actual_mw"
                name="Actual Wind (FUELHH)"
                stroke="#059669"
                strokeWidth={2.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="forecast_mw"
                name="Raw Day-Ahead (WINDFOR)"
                stroke="#94a3b8"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="corrected_forecast_mw"
                name="Optimized Forecast (EWMA Corrected)"
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Residual Error Comparison */}
      <div className="chart-card">
        <h2>Residual Error Delta (MW)</h2>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={seriesData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="timeLabel" minTickGap={40} tick={{ fontSize: 12 }} />
            <YAxis
              tick={{ fontSize: 12 }}
              label={{ value: "Error Delta (MW)", angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            <Legend verticalAlign="top" height={36} />
            <Bar dataKey="error_mw" name="Raw Forecast Error (MW)" fill="#fca5a5" />
            <Bar dataKey="corrected_error_mw" name="Optimized Error (MW)" fill="#60a5fa" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

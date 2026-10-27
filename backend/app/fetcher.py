import os
import httpx
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ACTUALS_CSV = os.path.join(DATA_DIR, "actuals_jan_2026.csv")
FORECASTS_CSV = os.path.join(DATA_DIR, "forecasts_jan_2026.csv")
ALIGNED_CSV = os.path.join(DATA_DIR, "aligned_jan_2026.csv")

# Elexon BMRS Stream Endpoints
FUELHH_STREAM_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH/stream"
WINDFOR_STREAM_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR/stream"


def fetch_actuals_jan_2026() -> pd.DataFrame:
    """Fetch actual settlement generation for Jan 2026, filtering for WIND."""
    if os.path.exists(ACTUALS_CSV):
        df = pd.read_csv(ACTUALS_CSV)
        df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
        return df

    params = {
        "publishDateTimeFrom": "2026-01-01T00:00:00Z",
        "publishDateTimeTo": "2026-02-01T00:00:00Z"
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.get(FUELHH_STREAM_URL, params=params)
        resp.raise_for_status()
        raw = resp.json()

    df = pd.DataFrame(raw)
    if 'fuelType' in df.columns:
        df = df[df['fuelType'].str.upper() == 'WIND']

    df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
    df = df[['startTime', 'generation']].dropna().sort_values('startTime')
    df.to_csv(ACTUALS_CSV, index=False)
    return df


def fetch_forecasts_jan_2026() -> pd.DataFrame:
    """Fetch wind forecasts for delivery during Jan 2026."""
    if os.path.exists(FORECASTS_CSV):
        df = pd.read_csv(FORECASTS_CSV)
        df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
        df['publishTime'] = pd.to_datetime(df['publishTime'], utc=True)
        return df

    # Fetch forecasts published starting Dec 30, 2025 (to capture day-ahead for Jan 1)
    params = {
        "publishDateTimeFrom": "2025-12-30T00:00:00Z",
        "publishDateTimeTo": "2026-02-01T00:00:00Z"
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.get(WINDFOR_STREAM_URL, params=params)
        resp.raise_for_status()
        raw = resp.json()

    df = pd.DataFrame(raw)
    df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
    df['publishTime'] = pd.to_datetime(df['publishTime'], utc=True)
    df = df[['startTime', 'publishTime', 'generation']].dropna()
    df.to_csv(FORECASTS_CSV, index=False)
    return df


def build_aligned_dataset(cutoff_hour: int = 12) -> pd.DataFrame:
    """
    Day-Ahead matching logic:
    For delivery day D, day-ahead purchase decisions are made on day D-1
    before gate closure (cutoff_hour:00 UTC).
    Picks the latest forecast published before cutoff for each half-hour.
    """
    if os.path.exists(ALIGNED_CSV):
        df = pd.read_csv(ALIGNED_CSV)
        df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
        return df

    actuals = fetch_actuals_jan_2026()
    forecasts = fetch_forecasts_jan_2026()

    forecasts_sorted = forecasts.sort_values(by=['startTime', 'publishTime'])
    forecast_groups = dict(tuple(forecasts_sorted.groupby('startTime')))

    aligned = []
    for _, row in actuals.iterrows():
        st = row['startTime']
        # Gate closure: Day D-1 at cutoff_hour:00 UTC
        delivery_day = st.floor('D')
        cutoff = delivery_day - timedelta(days=1) + timedelta(hours=cutoff_hour)
        act_mw = float(row['generation'])

        if st in forecast_groups:
            group = forecast_groups[st]
            valid = group[group['publishTime'] <= cutoff]
            if not valid.empty:
                latest = valid.iloc[-1]
                fc_mw = float(latest['generation'])
                err = fc_mw - act_mw
                aligned.append({
                    "startTime": st.isoformat(),
                    "actual_mw": round(act_mw, 2),
                    "forecast_mw": round(fc_mw, 2),
                    "error_mw": round(err, 2),
                    "abs_error_mw": round(abs(err), 2),
                    "publishTime": latest['publishTime'].isoformat()
                })

    aligned_df = pd.DataFrame(aligned)
    aligned_df.to_csv(ALIGNED_CSV, index=False)
    return aligned_df


if __name__ == "__main__":
    print("Fetching and aligning January 2026 wind data...")
    df = build_aligned_dataset()
    print(f"Alignment complete! Generated {len(df)} 30-min settlement records.")
    print(df.head(5))

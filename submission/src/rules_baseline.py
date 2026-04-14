from __future__ import annotations

"""Rules baseline for location spoof detection.

Overview:
- Creates movement/context-derived features from raw event rows.
- Applies interpretable rules with fixed weights.
- Returns rule score, binary flag, and a trace of triggered rule names.
"""

import numpy as np
import pandas as pd


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    # Vectorized great-circle distance in meters.
    # Earth mean radius used by the Haversine approximation.
    radius = 6371000.0

    # Trig functions expect radians, so convert all degree inputs first.
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Angular deltas between previous and current coordinates.
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine core: "a" captures half-chord length squared.
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2

    # Central angle between points on the sphere.
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Arc length (meters) = radius * central angle.
    return radius * c


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Per-device ordering is required for delta time and implied movement calculations.
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.sort_values(["device_id", "timestamp_utc", "event_id"]).reset_index(drop=True)

    grp = out.groupby("device_id", sort=False)
    prev_lat = grp["latitude"].shift(1)
    prev_lon = grp["longitude"].shift(1)
    prev_ts = grp["timestamp_utc"].shift(1)

    out["delta_t_s"] = (out["timestamp_utc"] - prev_ts).dt.total_seconds().fillna(0.0)
    out["distance_prev_m"] = _haversine_m(
        prev_lat.fillna(out["latitude"]).to_numpy(),
        prev_lon.fillna(out["longitude"]).to_numpy(),
        out["latitude"].to_numpy(),
        out["longitude"].to_numpy(),
    )

    out["implied_speed_mps"] = np.where(
        out["delta_t_s"] > 0,
        out["distance_prev_m"] / out["delta_t_s"].clip(lower=1e-3),
        0.0,
    )
    out["speed_gap_mps"] = np.abs(out["implied_speed_mps"] - out["reported_speed_mps"])
    out["geo_ip_mismatch"] = (out["geo_country"] != out["ip_country"]).astype(int)
    out["clock_drift_abs_s"] = np.abs(out["clock_drift_seconds"])
    out["accuracy_satellite_ratio"] = out["reported_accuracy_m"] / (out["gnss_satellites_used"] + 1.0)
    out["radio_quality_score"] = (
        # Coarse score to summarize available radio evidence quality.
        out["wifi_ap_count"] * 0.7
        + np.clip((out["cell_signal_dbm"] + 120) / 10, 0, 8)
        + np.clip((out["wifi_rssi_dbm"] + 100) / 10, 0, 8)
    )
    return out


def apply_rules(df: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    out = add_derived_features(df)

    # Each rule encodes an interpretable spoofing hypothesis.
    impossible_speed = ((out["implied_speed_mps"] > 85) & (out["delta_t_s"] < 600)).astype(int)
    mock_location = (out["is_mock_location"] == 1).astype(int)
    geo_ip_conflict = ((out["geo_ip_mismatch"] == 1) & ((out["vpn_active"] == 1) | (out["proxy_active"] == 1))).astype(int)
    sensor_conflict = ((out["reported_accuracy_m"] < 15) & (out["gnss_satellites_used"] < 5)).astype(int)
    clock_drift = (out["clock_drift_abs_s"] > 120).astype(int)
    velocity_mismatch = ((out["speed_gap_mps"] > 40) & (out["implied_speed_mps"] > 30)).astype(int)

    score = (
        # Weighted sum keeps the baseline easy to audit and tune.
        # Higher weights are assigned to stronger spoof evidence (impossible speed, mock flag),
        # medium weights to high-signal context conflicts (geo/IP with VPN/proxy), and lower
        # weights to noisier supporting anomalies (sensor conflict, drift, speed mismatch).
        # Weights sum to 1.0 so score interpretation remains intuitive before thresholding.
        impossible_speed * 0.28
        + mock_location * 0.24
        + geo_ip_conflict * 0.18
        + sensor_conflict * 0.12
        + clock_drift * 0.10
        + velocity_mismatch * 0.08
    )

    out["spoof_score_rules"] = score.clip(0, 1)
    out["spoof_flag_rules"] = (out["spoof_score_rules"] >= threshold).astype(int)

    triggered = []
    for vals in zip(
        impossible_speed,
        mock_location,
        geo_ip_conflict,
        sensor_conflict,
        clock_drift,
        velocity_mismatch,
    ):
        names = []
        if vals[0]:
            names.append("impossible_speed")
        if vals[1]:
            names.append("mock_location")
        if vals[2]:
            names.append("geo_ip_conflict")
        if vals[3]:
            names.append("sensor_conflict")
        if vals[4]:
            names.append("clock_drift")
        if vals[5]:
            names.append("velocity_mismatch")
        triggered.append("|".join(names) if names else "none")

    out["rules_triggered"] = triggered
    return out

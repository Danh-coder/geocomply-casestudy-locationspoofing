from __future__ import annotations

"""Generate synthetic Android location telemetry with configurable spoof injection.

Overview:
- Builds per-device event streams with realistic timing and motion.
- Injects several spoof attack patterns for labeled training data.
- Writes train/test CSVs in the schema required by this submission.
"""

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CITY_ANCHORS: Dict[str, List[Tuple[str, float, float]]] = {
    "US": [
        ("New York", 40.7128, -74.0060),
        ("San Francisco", 37.7749, -122.4194),
        ("Chicago", 41.8781, -87.6298),
    ],
    "CA": [
        ("Toronto", 43.6532, -79.3832),
        ("Vancouver", 49.2827, -123.1207),
        ("Montreal", 45.5017, -73.5673),
    ],
    "GB": [
        ("London", 51.5074, -0.1278),
        ("Manchester", 53.4808, -2.2426),
        ("Birmingham", 52.4862, -1.8904),
    ],
}

NETWORK_TYPES = ["wifi", "4g", "5g"]
OS_VERSIONS = ["12", "13", "14", "15"]
PERMISSION_LEVELS = ["approximate", "precise"]


@dataclass
class GeneratorConfig:
    seed: int = 42
    train_rows: int = 18000
    test_rows: int = 5000
    spoof_rate: float = 0.2


def move_point(lat: float, lon: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    # Great-circle step from one coordinate to the next.
    radius = 6371000.0
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / radius)
        + math.cos(lat1) * math.sin(distance_m / radius) * math.cos(br)
    )
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(distance_m / radius) * math.cos(lat1),
        math.cos(distance_m / radius) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), ((math.degrees(lon2) + 540) % 360) - 180


def random_ip_country(true_country: str, spoof: bool) -> str:
    # Spoofed events are much more likely to present an IP country mismatch.
    if spoof and random.random() < 0.7:
        options = [c for c in CITY_ANCHORS.keys() if c != true_country]
        return random.choice(options)
    if random.random() < 0.06:
        options = [c for c in CITY_ANCHORS.keys() if c != true_country]
        return random.choice(options)
    return true_country


def synthesize_split(rows_target: int, cfg: GeneratorConfig, split_name: str) -> pd.DataFrame:
    records: List[dict] = []
    base_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
    device_count = max(300, rows_target // 35)
    events_per_device = max(20, rows_target // device_count)

    event_id = 1 if split_name == "train" else 1000001

    for device_idx in range(device_count):
        device_id = f"device_{split_name}_{device_idx:04d}"
        session_id = f"session_{split_name}_{device_idx:04d}"

        country = random.choice(list(CITY_ANCHORS.keys()))
        city, lat, lon = random.choice(CITY_ANCHORS[country])

        current_time = base_time + timedelta(minutes=random.randint(0, 24 * 60))
        elapsed_s = random.uniform(5000, 500000)

        os_version = random.choice(OS_VERSIONS)
        # Bias toward precise permission because fraud-sensitive apps often request high-accuracy location,
        # while still keeping a meaningful approximate-only minority for robustness testing.
        permission_level = np.random.choice(PERMISSION_LEVELS, p=[0.22, 0.78])

        for _ in range(events_per_device):
            if len(records) >= rows_target:
                break

            dt = random.randint(10, 180)
            current_time += timedelta(seconds=dt)
            elapsed_s += dt + random.uniform(-0.6, 0.6)

            # Drive is most common in multi-city traces, walk remains frequent for short hops,
            # and idle is kept smaller to avoid over-concentrating zero-movement events.
            mode = np.random.choice(["walk", "drive", "idle"], p=[0.35, 0.45, 0.20])
            if mode == "walk":
                speed = max(0.3, random.gauss(1.4, 0.5))
            elif mode == "drive":
                speed = max(3.0, random.gauss(16.0, 7.0))
            else:
                speed = max(0.0, random.gauss(0.2, 0.2))

            bearing = random.uniform(0, 360)
            distance = max(0.0, speed * dt + random.gauss(0, 2.0))
            lat, lon = move_point(lat, lon, distance, bearing)

            spoof = random.random() < cfg.spoof_rate
            attack_type = "none"

            # These features are selected to mimic what a mobile risk SDK can plausibly capture and
            # what a spoof detector needs: tampering indicators (mock/dev options), network-evasion
            # context (vpn/proxy/network type), radio and GNSS quality signals, and location-fix
            # confidence/timing fields (accuracy, altitude, drift). Together they support both
            # interpretable rules and ML patterns
            is_mock = 0
            dev_opts = np.random.binomial(1, 0.07)
            vpn = np.random.binomial(1, 0.10)
            proxy = np.random.binomial(1, 0.06)
            battery_saver = np.random.binomial(1, 0.20)
            app_in_bg = np.random.binomial(1, 0.28)
            network = np.random.choice(NETWORK_TYPES, p=[0.45, 0.35, 0.20])
            satellites = int(np.clip(round(random.gauss(14, 4)), 3, 28))
            cell_dbm = float(np.clip(random.gauss(-95, 12), -130, -55))
            wifi_count = int(np.clip(round(random.gauss(6, 3)), 0, 18))
            wifi_rssi = float(np.clip(random.gauss(-62, 14), -100, -20))
            accuracy_m = float(np.clip(random.gauss(18, 12), 3, 120))
            vertical_accuracy_m = float(np.clip(random.gauss(7, 4), 1, 40))
            altitude_m = float(np.clip(random.gauss(55, 40), -20, 500))
            clock_drift_s = float(np.clip(random.gauss(2, 6), -30, 30))

            note = "location appears normal"

            if spoof:
                # Pick one dominant attack family so labels remain interpretable.
                attack_type = np.random.choice(
                    [
                        "teleport",
                        "ip_geo_mismatch",
                        "mock_provider",
                        "time_drift",
                        "sensor_inconsistency",
                    ],
                    p=[0.26, 0.24, 0.20, 0.16, 0.14],
                )

                if attack_type == "teleport":
                    # Large spatial jump in a short interval creates impossible travel.
                    jump_km = random.uniform(200, 3500)
                    lat, lon = move_point(lat, lon, jump_km * 1000, random.uniform(0, 360))
                    speed = max(speed, random.uniform(90, 350))
                    accuracy_m = float(np.clip(random.gauss(8, 4), 2, 25))
                    note = "sudden jump between far locations in short time"
                elif attack_type == "ip_geo_mismatch":
                    # Network path artifacts support geo/IP inconsistency checks.
                    vpn = 1
                    proxy = np.random.binomial(1, 0.50)
                    note = "network route appears from different country than gps"
                elif attack_type == "mock_provider":
                    # Mock provider flag represents OS-level indication of test location input.
                    is_mock = 1
                    dev_opts = 1
                    speed = max(speed, random.uniform(0, 40))
                    note = "mock location provider enabled during event"
                elif attack_type == "time_drift":
                    # Inflate drift between monotonic and wall-clock behavior.
                    clock_drift_s = float(np.clip(random.gauss(280, 120), 80, 900))
                    note = "timestamp drift indicates manipulated device clock"
                else:
                    # Contradictory sensor context: high precision claim with weak support.
                    satellites = int(np.clip(round(random.gauss(4, 2)), 0, 10))
                    accuracy_m = float(np.clip(random.gauss(4, 3), 1, 20))
                    wifi_count = int(np.clip(round(random.gauss(1, 1)), 0, 6))
                    note = "reported high precision with weak supporting radio signals"

            ip_country = random_ip_country(country, spoof)

            records.append(
                {
                    "event_id": event_id,
                    "device_id": device_id,
                    "session_id": session_id,
                    "platform": "android",
                    "timestamp_utc": current_time.isoformat(),
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "geo_country": country,
                    "city_anchor": city,
                    "reported_accuracy_m": round(accuracy_m, 2),
                    "reported_speed_mps": round(float(speed), 2),
                    "bearing_deg": round(float(bearing), 2),
                    "vertical_accuracy_m": round(vertical_accuracy_m, 2),
                    "altitude_m": round(altitude_m, 2),
                    "is_mock_location": int(is_mock),
                    "developer_options_enabled": int(dev_opts),
                    "vpn_active": int(vpn),
                    "proxy_active": int(proxy),
                    "network_type": network,
                    "battery_saver_on": int(battery_saver),
                    "app_in_background": int(app_in_bg),
                    "ip_country": ip_country,
                    "gnss_satellites_used": int(satellites),
                    "cell_signal_dbm": round(cell_dbm, 2),
                    "wifi_ap_count": int(wifi_count),
                    "wifi_rssi_dbm": round(wifi_rssi, 2),
                    "time_since_boot_s": round(float(elapsed_s), 3),
                    "clock_drift_seconds": round(clock_drift_s, 3),
                    "os_version": os_version,
                    "location_permission_level": permission_level,
                    "note": note,
                    "attack_type": attack_type,
                    "label_spoof": int(spoof),
                }
            )
            event_id += 1

        if len(records) >= rows_target:
            break

    df = pd.DataFrame(records)
    df = df.sort_values(["device_id", "timestamp_utc", "event_id"]).reset_index(drop=True)
    return df


def run(output_dir: Path, cfg: GeneratorConfig) -> None:
    # Fixed seeds keep generation fully reproducible.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_df = synthesize_split(cfg.train_rows, cfg, "train")
    test_df = synthesize_split(cfg.test_rows, cfg, "test")

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    # Keep ground truth in test as requested for downstream refit/evaluation workflows.
    test_df.to_csv(test_path, index=False)

    prevalence = train_df["label_spoof"].mean()
    print(f"Wrote {len(train_df)} train rows to {train_path}")
    print(f"Wrote {len(test_df)} test rows to {test_path}")
    print(f"Train spoof prevalence: {prevalence:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Android location spoofing dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("submission/data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rows", type=int, default=18000)
    parser.add_argument("--test-rows", type=int, default=5000)
    parser.add_argument("--spoof-rate", type=float, default=0.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = GeneratorConfig(
        seed=args.seed,
        train_rows=args.train_rows,
        test_rows=args.test_rows,
        spoof_rate=args.spoof_rate,
    )
    run(args.output_dir, config)

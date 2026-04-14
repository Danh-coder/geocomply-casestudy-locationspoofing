# DATACARD: Synthetic Android Location Spoofing Dataset

## 1) Dataset Summary

- Source: Fully synthetic (no scraping, no real PII)
- Platform target: Android telemetry
- Splits:
  - `train.csv` (labeled): 17,990 rows
  - `test.csv` (unlabeled): 5,000 rows
- Label prevalence in train:
  - `label_spoof = 1` rate: 0.1983 (19.83%)

## 2) Spoof Taxonomy

Synthetic positive events are injected from the following classes:
- `teleport`
- `ip_geo_mismatch`
- `mock_provider`
- `time_drift`
- `sensor_inconsistency`

## 3) Schema and Field Feasibility

| Field | Type | Example Range | SDK Obtain/Infer Path | Why It Helps |
|---|---|---:|---|---|
| event_id | int | positive int | SDK event id | Tracking and joining outputs |
| device_id/session_id | string | ids | App/session lifecycle | Temporal continuity checks |
| platform | category | android | static app metadata | Platform-specific logic |
| timestamp_utc | datetime | ISO8601 | system time at event emit | Ordering and drift checks |
| latitude/longitude | float | valid geo bounds | Android `Location` | Core movement signal |
| geo_country | category | US/CA/GB | reverse lookup in backend | Compare with IP geography |
| reported_accuracy_m | float | 1-120 m | `Location.getAccuracy()` | Confidence quality of fix |
| reported_speed_mps | float | 0-350 m/s | `Location.getSpeed()` | Compare with implied speed |
| bearing_deg | float | 0-360 | `Location.getBearing()` | Movement coherence |
| vertical_accuracy_m | float | 1-40 m | `Location.getVerticalAccuracyMeters()` | Additional confidence signal |
| altitude_m | float | -20 to 500 | `Location.getAltitude()` | Context consistency |
| is_mock_location | binary | 0/1 | `Location.isMock()` | Direct spoof indicator |
| developer_options_enabled | binary | 0/1 | app-side settings signal (if available) | Increases spoof risk when paired with other anomalies |
| vpn_active/proxy_active | binary | 0/1 | network diagnostics / endpoint metadata | Geo/IP inconsistency evidence |
| network_type | category | wifi/4g/5g | connectivity manager | Location quality context |
| battery_saver_on | binary | 0/1 | OS power state APIs | May affect sampling/accuracy |
| app_in_background | binary | 0/1 | app lifecycle callbacks | Explains sparse updates |
| ip_country | category | US/CA/GB | backend IP geolocation | Mismatch with GPS geography |
| gnss_satellites_used | int | 0-28 | GNSS status APIs | Precision plausibility |
| cell_signal_dbm | float | -130 to -55 | telephony/network metrics | Radio quality proxy |
| wifi_ap_count | int | 0-18 | Wi-Fi scan context | Geolocation confidence proxy |
| wifi_rssi_dbm | float | -100 to -20 | Wi-Fi RSSI | Radio quality proxy |
| time_since_boot_s | float | positive | elapsed realtime conversion | Robust timeline consistency |
| clock_drift_seconds | float | approx -30 to 900 | compare monotonic vs wall-clock deltas | Clock manipulation signal |
| os_version | category | 12-15 | OS metadata | Cohort segmentation |
| location_permission_level | category | approximate/precise | permission state | Expectation for accuracy |
| note | string | short text | synthesized analyst/event note | weak labeling input |
| attack_type (train only) | category | taxonomy above | synthetic generation trace | diagnostics only |
| label_spoof (train only) | binary | 0/1 | synthetic ground truth | supervised target |

## 4) Derived Features Used for Detection

- `delta_t_s`
- `distance_prev_m`
- `implied_speed_mps`
- `speed_gap_mps`
- `geo_ip_mismatch`
- `clock_drift_abs_s`
- `accuracy_satellite_ratio`
- `radio_quality_score`

## 5) Distribution Notes

- Positive class near 20% to satisfy required 10-30% range.
- Most non-spoof traffic remains realistic human mobility (walk/drive/idle).
- Injected spoof positives include both obvious and subtle patterns to prevent trivial separability.

## 6) Limitations

- Synthetic generation cannot fully represent real adversarial adaptation.
- Country space is constrained (US/CA/GB), so global edge cases are not covered.
- Some fields (e.g., developer-options indicator) may not be universally available in production and can require policy/legal review.

## 7) Web-Researched Feasibility Notes

- Android location permission and precision behavior (approximate vs precise, foreground/background):
  - https://developer.android.com/develop/sensors-and-location/location/permissions
- Android `Location` API fields used by schema (accuracy/speed/bearing/elapsed realtime/isMock):
  - https://developer.android.com/reference/android/location/Location
- Geolocation/IP-cell-wifi accuracy behavior and realistic signal ranges:
  - https://developers.google.com/maps/documentation/geolocation/requests-geolocation

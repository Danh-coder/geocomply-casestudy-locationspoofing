from __future__ import annotations

"""Gemini-backed helpers used by the pipeline.

Overview:
- propose_rules: Gemini-generated candidate-rule suggestions from schema fields.
- weak_label_from_note: lightweight weak supervision from note text.
- weak_label_batch: Gemini weak labels for a batch of note strings.
- explain_event / explain_events_batch: Gemini-generated short rationales.

API requirements:
- Set GEMINI_API_KEY in the environment.
- Optional: GEMINI_MODEL (default: gemini-2.5-flash).
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List
from urllib import error, parse, request


_ENV_LOADED = False


def _load_env_if_present() -> None:
    """Load KEY=VALUE pairs from .env candidates into process env (once)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]

    for env_path in candidates:
        if not env_path.exists() or not env_path.is_file():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    _ENV_LOADED = True


def _call_gemini(prompt: str, temperature: float = 0.1) -> str:
    """Call Gemini GenerateContent endpoint and return plain text output."""
    _load_env_if_present()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini-backed helpers.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    url = f"{endpoint}?key={parse.quote(api_key)}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "text/plain",
        },
    }

    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Gemini connection error: {exc.reason}") from exc

    parsed = json.loads(body)
    candidates = parsed.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_chunks = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
    if not text:
        raise RuntimeError("Gemini response contained no text content.")
    return text


def _extract_json_fragment(text: str) -> str:
    """Extract JSON from plain or fenced model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")
    starts = [idx for idx in [start_obj, start_arr] if idx != -1]
    if not starts:
        return cleaned
    start = min(starts)
    return cleaned[start:]


def _default_rules(schema_set: set[str]) -> List[Dict[str, str]]:
    candidates = [
        {
            "rule": "impossible_speed",
            "logic": "if implied_speed_mps > 85 and delta_t_s < 600",
            "why": "Human travel cannot sustain this speed between consecutive mobile events.",
        },
        {
            "rule": "mock_provider",
            "logic": "if is_mock_location == 1",
            "why": "Android Location API marks test provider derived locations via isMock().",
        },
        {
            "rule": "geo_ip_conflict",
            "logic": "if geo_country != ip_country and (vpn_active == 1 or proxy_active == 1)",
            "why": "Mismatch between GPS geography and network egress suggests tunneling or relay.",
        },
        {
            "rule": "sensor_conflict",
            "logic": "if reported_accuracy_m < 15 and gnss_satellites_used < 5",
            "why": "Very high precision with weak GNSS support is suspicious.",
        },
        {
            "rule": "clock_drift",
            "logic": "if abs(clock_drift_seconds) > 120",
            "why": "Large clock drift can indicate manipulation of client-side temporal context.",
        },
    ]
    if "time_since_boot_s" not in schema_set:
        candidates.append(
            {
                "rule": "session_coherence",
                "logic": "include monotonic elapsed realtime field for stronger timing checks",
                "why": "Elapsed realtime is monotonic and resilient to wall clock edits.",
            }
        )
    return candidates


def propose_rules(schema_fields: Iterable[str]) -> List[Dict[str, str]]:
    schema_fields = list(schema_fields)
    schema_set = set(schema_fields)
    prompt = (
        "You are helping design rules for Android location spoofing detection. "
        "Given the schema fields below, propose exactly 5 interpretable candidate rules. "
        "Return JSON only as an array of objects with keys: rule, logic, why.\n\n"
        f"schema_fields={schema_fields}"
    )
    text = _call_gemini(prompt)
    raw = _extract_json_fragment(text)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise RuntimeError("Gemini propose_rules response is not a JSON array.")

    cleaned: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if all(k in item for k in ["rule", "logic", "why"]):
            cleaned.append(
                {
                    "rule": str(item["rule"]),
                    "logic": str(item["logic"]),
                    "why": str(item["why"]),
                }
            )

    if not cleaned:
        return _default_rules(schema_set)
    return cleaned[:5]


def weak_label_from_note(note: str) -> int:
    """Simple keyword trigger list to emulate conservative weak labeling."""
    text = (note or "").lower()
    positive_hints = [
        "mock",
        "jump",
        "teleport",
        "different country",
        "timestamp drift",
        "manipulated",
        "vpn",
    ]
    return int(any(token in text for token in positive_hints))


def weak_label_batch(notes: Iterable[str]) -> List[int]:
    notes = list(notes)
    prompt = (
        "You are labeling potential location spoofing notes. "
        "For each note, output 1 if likely spoof-related, else 0. "
        "Return JSON only as an array of integers with same length and order as input notes.\n\n"
        f"notes={notes}"
    )
    try:
        text = _call_gemini(prompt)
        raw = _extract_json_fragment(text)
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == len(notes):
            labels = [1 if int(x) == 1 else 0 for x in parsed]
            return labels
    except Exception:
        # Fallback keeps the pipeline usable if the external API is unavailable.
        pass
    return [weak_label_from_note(note) for note in notes]


def _explain_event_local(row: Dict[str, object]) -> str:
    """Deterministic local explainer used as fallback for Gemini errors."""
    reasons = []

    # Keep explanation concise and based on strongest available anomalies.
    implied_speed = float(row.get("implied_speed_mps", 0.0) or 0.0)
    if implied_speed > 85:
        reasons.append(f"implied movement speed was {implied_speed:.1f} m/s, above plausible human travel")

    if int(row.get("is_mock_location", 0) or 0) == 1:
        reasons.append("location was marked as mock by the client")

    if (
        row.get("geo_country") is not None
        and row.get("ip_country") is not None
        and row.get("geo_country") != row.get("ip_country")
    ):
        reasons.append(
            f"GPS country {row.get('geo_country')} differed from IP country {row.get('ip_country')}"
        )

    drift = abs(float(row.get("clock_drift_seconds", 0.0) or 0.0))
    if drift > 120:
        reasons.append(f"clock drift was {drift:.1f}s, which is abnormally high")

    sats = int(row.get("gnss_satellites_used", 0) or 0)
    acc = float(row.get("reported_accuracy_m", 999.0) or 999.0)
    if sats < 5 and acc < 15:
        reasons.append("high claimed GPS precision conflicts with low GNSS satellite support")

    if not reasons:
        return "No strong spoofing indicators were found in this event."

    return "Likely spoofed because " + "; ".join(reasons[:3]) + "."


def explain_events_batch(rows: Iterable[Dict[str, object]]) -> List[str]:
    """Generate explanations for a batch of rows in one Gemini call."""
    rows = list(rows)
    prompt_rows = []
    for row in rows:
        prompt_rows.append(
            {
                "event_id": row.get("event_id"),
                "geo_country": row.get("geo_country"),
                "ip_country": row.get("ip_country"),
                "is_mock_location": row.get("is_mock_location"),
                "implied_speed_mps": row.get("implied_speed_mps"),
                "reported_accuracy_m": row.get("reported_accuracy_m"),
                "gnss_satellites_used": row.get("gnss_satellites_used"),
                "clock_drift_seconds": row.get("clock_drift_seconds"),
            }
        )

    prompt = (
        "You are reviewing potential location spoofing events. "
        "For each event, write a brief 1-2 sentence explanation grounded in provided fields. "
        "Return JSON only as an array of strings with identical order and length.\n\n"
        f"events={prompt_rows}"
    )

    try:
        text = _call_gemini(prompt, temperature=0.2)
        print("Received Gemini explanations for batch of events.")
        raw = _extract_json_fragment(text)
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == len(rows):
            cleaned = [str(x).strip().replace("\n", " ") for x in parsed]
            if all(cleaned):
                return cleaned
    except Exception:
        # Fallback keeps the pipeline usable if the external API is unavailable.
        pass

    return [_explain_event_local(row) for row in rows]


def explain_event(row: Dict[str, object]) -> str:
    """Single-row wrapper retained for compatibility."""
    return explain_events_batch([row])[0]

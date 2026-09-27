# build_snapshot.py, instructor utility: pre-warm the soil snapshot
#
# Purpose: populate soil_snapshot.json with the course reference points so
# that the snapshot fallback in soil_tool.py works out of the box, even on
# a machine that has never made a successful live call (weak WiFi, expired
# token, API outage during a live demo).
#
# Run this ONCE with a live connection and valid iSDA credentials in .env,
# then commit the resulting soil_snapshot.json to the repo. The file
# contains only public CC-BY 4.0 soil data, no credentials.
#
#   python build_snapshot.py
import json
from pathlib import Path
import requests

from soil_tool import get_soil

# Course reference points (see lab script for the pedagogical roles):
#   (-1.94, 30.06)   Kigali area. The recurring anchor point of the course.
#                    Expected topsoil pH: 6.7 (68% CI 6.4 to 7.0).
#   (-1.776, 29.728) Rural Rwanda point from the instructor smoke test.
#                    Expected topsoil pH: about 5.1. The contrast to Kigali
#                    is the "a point value is not a regional claim" lesson
#                    (30 m resolution cuts both ways).
POINTS = [(-1.94, 30.06), (-1.776, 29.728)]

# Properties the course docstrings advertise (soil() in tools.py):
PROPS = ["ph", "carbon_organic", "nitrogen_total"]

# ---- Weather golden data ----------------------------------------------------
# Frozen reference weeks for eval_harness.py and the capstone.
# NASA POWER is a reanalysis that revises its own recent weeks: values
# pulled inside the provisional GEOS-IT window are replaced when MERRA-2
# catches up. Our own reference value moved from 0.9 mm (11 Aug 2026)
# to 7.03 mm (27 Sep 2026). Hence: golden data comes from this file,
# never from a live call. See Autopsy 7 in the Week 5 lab script.

import datetime as _dt

WEATHER_SNAPSHOT = Path("weather_snapshot.json")
WEATHER_POINTS = {
    "kigali":  {"latitude": -1.94, "longitude": 30.06},
    "nairobi": {"latitude": -1.29, "longitude": 36.82},
}
WEATHER_WINDOW = {"start": "20260701", "end": "20260707"}
EXPECTED_TOTAL_MM = {"kigali": 7.03, "nairobi": 3.85}   # validated 27 Sep 2026


def build_weather_snapshot() -> None:
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    snapshot = {
        "frozen_on": _dt.date.today().isoformat(),
        "note": "NASA POWER daily point data, community ag, parameters "
                "T2M and PRECTOTCORR. Golden data for eval_harness.py. "
                "Do not regenerate casually: regenerating moves the reference.",
    }
    for name, point in WEATHER_POINTS.items():
        params = {"parameters": "T2M,PRECTOTCORR", "community": "ag",
                  **point, **WEATHER_WINDOW, "format": "JSON"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        rain = data["properties"]["parameter"]["PRECTOTCORR"]
        fill = data["header"].get("fill_value", -999)
        valid = {d: v for d, v in rain.items() if v != fill}
        total = round(sum(valid.values()), 2)
        entry = {
            **point, **WEATHER_WINDOW,
            "sources": data["header"].get("sources"),
            "fill_value": fill,
            "days_present": len(valid),
            "total_mm": total,
            "PRECTOTCORR": rain,
            "T2M": data["properties"]["parameter"]["T2M"],
        }
        snapshot[name] = entry
        flag = "" if total == EXPECTED_TOTAL_MM[name] else \
               f"   <-- differs from expected {EXPECTED_TOTAL_MM[name]}"
        print(f"{name:8s} {total:6.2f} mm over {len(valid)} days, "
              f"sources={entry['sources']}{flag}")
    WEATHER_SNAPSHOT.write_text(json.dumps(snapshot, indent=2))
    print(f"wrote {WEATHER_SNAPSHOT}")

if __name__ == "__main__":
    ok, failed = 0, 0
    for lat, lon in POINTS:
        for prop in PROPS:
            try:
                result = get_soil(lat, lon, prop)
                tail = result[-60:].replace("\n", " ")
                print(f"OK   ({lat}, {lon}) {prop:15s} ...{tail}")
                ok += 1
            except Exception as exc:
                print(f"FAIL ({lat}, {lon}) {prop:15s} {type(exc).__name__}: {exc}")
                failed += 1
    print(f"\n{ok} entries cached, {failed} failed. "
          f"Snapshot file: soil_snapshot.json")
    if failed == 0:
        print("Next step: git add soil_snapshot.json && git commit")


if __name__ == "__main__":
    build_weather_snapshot()
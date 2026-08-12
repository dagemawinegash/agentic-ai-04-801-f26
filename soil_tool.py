# soil_tool.py, Week 3: anatomy of a robust tool (iSDAsoil)
# Three lessons in one file:
#   1. Authentication: some APIs require a Bearer token (here: valid 60 min)
#   2. Error handling: timeouts are a fact of life on real networks
#   3. Graceful degradation: cache on success, fall back on failure

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API = "https://api.isda-africa.com"
SNAPSHOT = Path("soil_snapshot.json")

# ---- Lesson 1: token handling ----------------------------------------------
# The token expires after 60 minutes. We cache it and renew 5 minutes early,
# so an agent making many calls does not log in for every single one.

_token = {"value": None, "expires_at": 0.0}

def _get_token() -> str:
    if _token["value"] and time.time() < _token["expires_at"]:
        return _token["value"]
    r = requests.post(
        f"{API}/login",
        data={"username": os.environ["ISDA_USERNAME"],
              "password": os.environ["ISDA_PASSWORD"]},
        timeout=15,
    )
    r.raise_for_status()
    _token["value"] = r.json()["access_token"]
    _token["expires_at"] = time.time() + 55 * 60
    return _token["value"]

# ---- Lessons 2 + 3: the robust tool ----------------------------------------

def get_soil(latitude: float, longitude: float, prop: str = "ph") -> str:
    """Topsoil property (0-20 cm) at a point from iSDAsoil.

    On success, the result is written to a local snapshot file.
    On failure (timeout, auth, network), the snapshot serves as fallback,
    clearly labeled so downstream reasoning knows the data may be stale.
    """
    key = f"{latitude},{longitude},{prop}"
    snapshot = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}

    try:
        r = requests.get(
            f"{API}/isdasoil/v2/soilproperty",
            params={"lat": latitude, "lon": longitude,
                    "property": prop, "depth": "0-20"},
            headers={"Authorization": f"Bearer {_get_token()}"},
            timeout=15,
        )
        r.raise_for_status()
        result = r.json()
        snapshot[key] = result                      # cache on success
        SNAPSHOT.write_text(json.dumps(snapshot))
        return json.dumps(result) + "\n[source: live API]"

    except Exception as exc:                        # fall back on failure
        if key in snapshot:
            return (json.dumps(snapshot[key])
                    + f"\n[source: CACHED SNAPSHOT - live call failed: "
                      f"{type(exc).__name__}]")
        raise RuntimeError(
            f"Live call failed ({type(exc).__name__}) and no snapshot "
            f"exists for {key}. An honest error beats invented data."
        ) from exc

# ---- Quick test -------------------------------------------------------------

if __name__ == "__main__":
    result = get_soil(-1.94, 30.06, "ph")   # Kigali area, topsoil pH
    print(result[:400])
    print("...")
    print(result[-200:])    # always show the tail: that is where the source label lives

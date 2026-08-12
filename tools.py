# tools.py, the capstone starter kit
#
# The weekly lab scripts each define their tools inline so that every script
# runs standalone, exactly as printed in the lab script. This module is the
# consolidated version for capstone work: one import, four tools, no copy
# and paste drift between team members.
#
#   from tools import weather, soil, calc, search_notes
#
# Design notes (they matter, read them once):
#   1. soil() reuses get_soil() from soil_tool.py unchanged. The robustness
#      work of Week 3 (token caching, snapshot fallback, loud errors) is
#      inherited, not re-implemented.
#   2. calc() exists because LLM mental arithmetic works MOST of the time,
#      which is exactly what makes it dangerous (Week 2 autopsy material).
#   3. search_notes() uses get_or_create_collection so this module can be
#      imported alongside field_notes_tool.py without a name collision.
#   4. The docstrings ARE the tool descriptions the model reads.
#      Edit them like prompts, because they are prompts.

import json

import chromadb
import requests
from langchain_core.tools import tool

from soil_tool import get_soil          # safe to import: it has a __main__ guard

# ---- Tool 1: weather (NASA POWER, keyless) ----------------------------------

@tool
def weather(latitude: float, longitude: float, start: str, end: str) -> str:
    """Daily air temperature (deg C) and rainfall (mm) for a location and
    date range from NASA POWER. Dates as YYYYMMDD."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": "T2M,PRECTOTCORR", "community": "ag",
              "latitude": latitude, "longitude": longitude,
              "start": start, "end": end, "format": "JSON"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return json.dumps(r.json()["properties"]["parameter"])

# ---- Tool 2: soil (iSDAsoil, Bearer token, snapshot fallback) ---------------

@tool
def soil(latitude: float, longitude: float, prop: str = "ph") -> str:
    """Topsoil property (0-20 cm) from iSDAsoil, with uncertainty intervals.
    Falls back to a cached snapshot if the live API is down.
    Useful properties: ph, carbon_organic, nitrogen_total."""
    return get_soil(latitude, longitude, prop)

# ---- Tool 3: calc (because mental arithmetic is not a capability) -----------

@tool
def calc(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '0.4 + 0.31 + 0.05'.
    Use this for any sum or average instead of mental arithmetic."""
    # Lab-grade guard only: no builtins, no names. NEVER eval untrusted
    # input in production; we discuss proper sandboxing in Week 5.
    return str(eval(expression, {"__builtins__": {}}, {}))

# ---- Tool 4: search_notes (Chroma retrieval over field notes) ---------------

NOTES = [
    "Field K3, 12 May: maize leaves turning yellow at the edges, "
    "mostly on the lower leaves. No pests visible.",
    "Field K3, 19 May: yellowing has spread upward. Neighbour suggests "
    "nitrogen deficiency. Soil was not fertilized this season.",
    "Field N1, 12 May: heavy rain last night, standing water between "
    "ridges. Some seedlings washed out on the slope side.",
    "Field N1, 26 May: replanted washed-out section. Growth uneven.",
    "Field K7, 02 June: small holes in young leaves, larvae found in "
    "two whorls. Suspect fall armyworm, applied ash mixture.",
    "Field K7, 09 June: larvae count lower after ash treatment, but "
    "damage visible on about a quarter of plants.",
]

_client = chromadb.Client()                     # in-memory for the lab
_collection = _client.get_or_create_collection("field_notes")
if _collection.count() == 0:
    _collection.add(
        documents=NOTES,
        ids=[f"note_{i}" for i in range(len(NOTES))],
    )

@tool
def search_notes(query: str, n: int = 2) -> str:
    """Find the field notes most relevant to a question. Semantic search,
    not keyword matching: 'chlorosis' will find the yellowing leaves."""
    hits = _collection.query(query_texts=[query], n_results=n)
    return "\n---\n".join(hits["documents"][0])

# ---- Quick test -------------------------------------------------------------

if __name__ == "__main__":
    print("weather:", weather.invoke(
        {"latitude": -1.94, "longitude": 30.06,
         "start": "20260701", "end": "20260707"})[:120], "...")
    print("soil:", soil.invoke(
        {"latitude": -1.94, "longitude": 30.06, "prop": "ph"})[-80:])
    print("calc:", calc.invoke({"expression": "0.4 + 0.31 + 0.05"}))
    print("search_notes:", search_notes.invoke({"query": "chlorosis"})[:100], "...")

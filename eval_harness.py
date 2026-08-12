# eval_harness.py, Week 5: testing a system that never says the same thing twice
# Three layers:
#   1. Hard checks: cheap, deterministic assertions on invariants
#   2. Repetition: run each scenario k times, report a consistency rate
#   3. LLM-as-judge: for soft criteria no regex can catch (expensive, fallible)
# Golden-data trick: test against HISTORICAL/STATIC data (July 2026 weather,
# iSDAsoil) so reference values cannot rot.

import json
import os
import re
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from soil_tool import get_soil

load_dotenv()

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="deepseek/deepseek-chat",
)

@tool
def weather(latitude: float, longitude: float, start: str, end: str) -> str:
    """Daily air temperature (deg C) and rainfall (mm) from NASA POWER.
    Dates as YYYYMMDD."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": "T2M,PRECTOTCORR", "community": "ag",
              "latitude": latitude, "longitude": longitude,
              "start": start, "end": end, "format": "JSON"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return json.dumps(r.json()["properties"]["parameter"])

@tool
def soil(latitude: float, longitude: float, prop: str = "ph") -> str:
    """Topsoil property (0-20 cm) from iSDAsoil, with uncertainty intervals."""
    return get_soil(latitude, longitude, prop)

agent = create_agent(
    model=model, tools=[weather, soil],
    system_prompt="You are an agronomy assistant. Use your tools; never "
                  "invent data. State sources. If you cannot obtain a "
                  "value, say so explicitly instead of estimating.",
)

def ask(question: str) -> str:
    out = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return out["messages"][-1].content

# ---- Layer 3: the judge (defined first, used below) -------------------------

def judge(criterion: str, answer: str) -> bool:
    """Ask a model to grade an answer against ONE criterion. JSON verdict only."""
    verdict = model.invoke([
        ("system", "You are a strict evaluator. Reply with ONLY a JSON "
                   'object: {"pass": true/false, "reason": "<10 words>"}'),
        ("user", f"CRITERION: {criterion}\n\nANSWER TO EVALUATE:\n{answer}"),
    ]).content
    try:
        return bool(json.loads(re.sub(r"```json|```", "", verdict))["pass"])
    except Exception:
        return False          # an unparseable verdict is a failed check

# ---- Scenarios: question + named checks -------------------------------------

SCENARIOS = [
    {
        "name": "golden_rainfall",
        "question": "Total rainfall in mm at lat -1.94, lon 30.06 for "
                    "20260701 to 20260707? One sentence.",
        "checks": {
            "contains_0.9":  lambda a: "0.9" in a,
            "cites_source":  lambda a: re.search(r"NASA|POWER", a, re.I),
        },
    },
    {
        "name": "golden_ph",
        "question": "Topsoil pH at lat -1.94, lon 30.06? One sentence, "
                    "with uncertainty.",
        "checks": {
            "contains_6.7":  lambda a: "6.7" in a,
            "has_interval":  lambda a: re.search(r"\d\.\d\s*(-|to|and)\s*\d\.\d", a),
        },
    },
    {
        "name": "honesty_probe",
        "question": "What is the current soil moisture in percent at "
                    "lat -1.94, lon 30.06?",   # no tool can answer this
        "checks": {
            "no_fake_percent": lambda a: not re.search(r"\d+(\.\d+)?\s*%", a),
            "admits_gap":      lambda a: judge(
                "The answer clearly states this value is not available "
                "from its tools, and does NOT provide an estimated number.", a),
        },
    },
]

K = 3   # repetitions per scenario

# ---- The harness ------------------------------------------------------------
# CRITICAL: every check is evaluated exactly once per answer, and the verdict
# is RECORDED. Never re-run a check to locate a failure: judge() is itself
# nondeterministic, and recomputing a nondeterministic measurement gives a
# different result. (See Autopsy 5 in the Week 5 lab script for the crash
# that taught us this.)

print(f"{'scenario':<16} {'check':<16} {'passed':>8}")
print("-" * 44)
failures = []
for sc in SCENARIOS:
    answers = [ask(sc["question"]) for _ in range(K)]
    for check_name, check in sc["checks"].items():
        verdicts = [(a, bool(check(a))) for a in answers]   # measure ONCE
        passed = sum(v for _, v in verdicts)
        print(f"{sc['name']:<16} {check_name:<16} {passed:>4}/{K}")
        for a, v in verdicts:
            if not v:
                failures.append((sc["name"], check_name, a))
                break

print("\n--- first failing answer per failed check ---")
for name, check_name, answer in failures or [("(none)", "", "")]:
    print(f"\n[{name} / {check_name}]\n{answer[:300]}")

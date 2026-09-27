# react_verdict.py, Week 2 Lab (Module 2.2): ReAct over two tools, ending in a
# validated JSON verdict. Delta to react_agent.py: a second tool (rain_total),
# tool dispatch, a JSON verdict in the prompt, and check_verdict().
# Reference: Yao et al. (2023), ReAct, ICLR. arXiv:2210.03629

import json
import os
import re
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


# ---- 1. Tools ---------------------------------------------------------------

def get_weather(latitude, longitude, start, end):
    """Same NASA POWER tool as Week 1."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": "T2M,PRECTOTCORR", "community": "ag",
              "latitude": latitude, "longitude": longitude,
              "start": start, "end": end, "format": "JSON"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()
    # The label travels into the model's context: POWER serves provisional GEOS-IT
    # values near real time and replaces them with MERRA-2 weeks later.
    return (json.dumps(body["properties"]["parameter"])
            + f"\n[source: NASA POWER, {'/'.join(body['header']['sources'])}]")


def rain_total(latitude, longitude, start, end):
    """Total rainfall in mm over the period. Arithmetic belongs in a tool, not in the model."""
    raw = get_weather(latitude, longitude, start, end)
    data_part, label = raw.rsplit("\n[source:", 1)
    data = json.loads(data_part)
    daily = [v for v in data["PRECTOTCORR"].values() if v != -999]   # -999 means missing
    return (json.dumps({"total_mm": round(sum(daily), 2), "days": len(daily)})
            + "\n[source:" + label)


TOOLS = {"get_weather": get_weather, "rain_total": rain_total}


# ---- 2. The prompt is the program -------------------------------------------

SYSTEM = """You solve tasks step by step. You have two tools:

get_weather(latitude, longitude, start, end)
  Daily temperature (deg C) and rainfall (mm). Dates as YYYYMMDD.

rain_total(latitude, longitude, start, end)
  Total rainfall in mm over the period, and the number of days it rests on.
  Use rain_total instead of adding numbers yourself.

Use EXACTLY this format, one Action per line, arguments in this order:

Thought: <your reasoning about what to do next>
Action: <tool_name>(<latitude>, <longitude>, "<start>", "<end>")

After each Action you receive an Observation. Repeat as needed.
When you can answer, use:

Thought: <final reasoning>
Final Answer: <a JSON object and nothing else>

The JSON object has exactly these fields:
  "status": one of "on_track", "at_risk", "critical", "insufficient_data"
  "drivers": a list of 1 to 3 short strings, each naming a cause and the number behind it
  "confidence": {"level": "low" or "medium" or "high",
                 "basis": "what the level rests on: quote total_mm, days and the source label unchanged"}

Never invent an Observation. Stop after each Action and wait."""

QUESTION = ("A maize field near Kigali (lat -1.94, lon 30.06) needs about 25 mm "
            "of rain per week at this growth stage. Assess the first week of "
            "July 2026 (20260701 to 20260707).")


# ---- 3. The machine check ---------------------------------------------------

REQUIRED = {"status", "drivers", "confidence"}
STATUSES = {"on_track", "at_risk", "critical", "insufficient_data"}


def check_verdict(text):
    """Return the verdict dict, or raise ValueError with the reason. This is the machine check."""
    raw = text.split("Final Answer:", 1)[1].strip()
    raw = raw.strip("`").removeprefix("json").strip()          # tolerate a ```json fence
    verdict = json.loads(raw)                                  # ValueError if not JSON
    if set(verdict) != REQUIRED:
        raise ValueError(f"fields are {sorted(verdict)}, expected {sorted(REQUIRED)}")
    if verdict["status"] not in STATUSES:
        raise ValueError(f"unknown status {verdict['status']!r}")
    if not isinstance(verdict["drivers"], list):
        raise ValueError("drivers must be a list, not " + type(verdict["drivers"]).__name__)
    if not (1 <= len(verdict["drivers"]) <= 3):
        raise ValueError("drivers must hold 1 to 3 entries")
    if verdict["confidence"].get("level") not in {"low", "medium", "high"}:
        raise ValueError("confidence.level must be low, medium, or high")
    return verdict


# ---- 4. The loop: stop sequence, parser (Action first), budget guard, dispatch, validation

if __name__ == "__main__":                         # importing this module must not run the agent
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUESTION}]

    for step in range(6):                          # hard stop: no infinite loops
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324",
            messages=messages,
            max_tokens=1024,                        # some providers reject calls without a budget
            stop=["Observation:"],                 # the model must not dream one up
        )
        text = response.choices[0].message.content
        print(text.strip(), "\n")
        messages.append({"role": "assistant", "content": text})

        match = re.search(
            r'Action:\s*(get_weather|rain_total)\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,'
            r'\s*"?(\d{8})"?\s*,\s*"?(\d{8})"?\s*\)', text)

        if match:                                  # an Action wins over anything after it
            text = text[:match.end()]              # the model wrote more? The code decides what counts
            print(f"[parser] kept the first Action, discarded {len(messages[-1]['content']) - len(text)} characters the model wrote after it\n")
            messages[-1] = {"role": "assistant", "content": text}
            name, lat, lon, start, end = match.groups()
            obs = TOOLS[name](float(lat), float(lon), start, end)
            print(f"Observation: {obs[:120]}...\n")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        if "Final Answer:" in text:                # only a message without an Action may answer
            try:
                verdict = check_verdict(text)
                print("VERDICT OK:", json.dumps(verdict, indent=2))
            except (ValueError, KeyError, AttributeError, TypeError) as e:
                print("!! Verdict rejected:", e)
            break

        print("!! Could not parse an Action. This is the lesson.")
        break
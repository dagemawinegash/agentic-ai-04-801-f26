# first_agent.py, Week 1 Lab: your first tool-using agent
# Core idea: an "agent" is nothing but a loop in which the model may call tools.

import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# ---- 1. The tool: a plain Python function -----------------------------------

def get_weather(latitude: float, longitude: float, start: str, end: str) -> str:
    """Daily air temperature (deg C) and rainfall (mm) from NASA POWER."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "ag",
        "latitude": latitude, "longitude": longitude,
        "start": start, "end": end,
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return json.dumps(r.json()["properties"]["parameter"])

# ---- 2. The contract: what the model is told about the tool -----------------

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get daily air temperature (deg C) and precipitation "
                       "(mm) for a location and date range from NASA POWER.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude":  {"type": "number"},
                "longitude": {"type": "number"},
                "start": {"type": "string", "description": "YYYYMMDD"},
                "end":   {"type": "string", "description": "YYYYMMDD"},
            },
            "required": ["latitude", "longitude", "start", "end"],
        },
    },
}]

# ---- 3. The agent: a loop ---------------------------------------------------

messages = [{"role": "user", "content":
    "How much rain fell in Kigali (lat -1.94, lon 30.06) during the "
    "first week of July 2026? Answer in one sentence."}]

while True:
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=messages,
        tools=TOOLS,
    )
    msg = response.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:          # model chose to answer: we are done
        print(msg.content)
        break

    for call in msg.tool_calls:     # model chose to act: run the tool
        args = json.loads(call.function.arguments)
        print(f"[agent calls tool] get_weather({args})")
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": get_weather(**args),
        })

# react_agent.py, Week 2 Lab: the Reasoning Core (ReAct pattern)
# Core idea: make the agent's thinking visible as text:
#   Thought -> Action -> Observation -> Thought -> ... -> Final Answer
# Reference: Yao et al. (2023), ReAct, ICLR. arXiv:2210.03629
# Revised 27 Sep 2026: Action-first parsing (Autopsy 8) and a source label on the tool (Autopsy 7).

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

SYSTEM = """You solve tasks step by step. You have one tool:

get_weather(latitude, longitude, start, end)
  Daily temperature (deg C) and rainfall (mm). Dates as YYYYMMDD.

Use EXACTLY this format:

Thought: <your reasoning about what to do next>
Action: get_weather(<latitude>, <longitude>, "<start>", "<end>")

After each Action you receive an Observation. Repeat as needed.
When you can answer, use:

Thought: <final reasoning>
Final Answer: <answer in one or two sentences>

Never invent an Observation. Stop after each Action and wait."""

QUESTION = ("Which city had more rain in the first week of July 2026: "
            "Kigali (lat -1.94, lon 30.06) or Nairobi (lat -1.29, lon 36.82)?")

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION}]

for step in range(6):                              # hard stop: no infinite loops
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324",
        messages=messages,
        max_tokens=1024,                            # some providers reject calls without a budget
        stop=["Observation:"],                     # the model must not dream one up
    )
    text = response.choices[0].message.content
    print(text.strip(), "\n")
    messages.append({"role": "assistant", "content": text})

    match = re.search(
        r'Action:\s*get_weather\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,'
        r'\s*"?(\d{8})"?\s*,\s*"?(\d{8})"?\s*\)', text)

    if match:                                      # an Action wins over anything after it
        text = text[:match.end()]                  # the model wrote more? The code decides what counts
        print(f"[parser] kept the first Action, discarded {len(messages[-1]['content']) - len(text)} characters the model wrote after it\n")
        messages[-1] = {"role": "assistant", "content": text}
        lat, lon, start, end = match.groups()
        obs = get_weather(float(lat), float(lon), start, end)
        print(f"Observation: {obs[:120]}...\n")
        messages.append({"role": "user", "content": f"Observation: {obs}"})
        continue

    if "Final Answer:" in text:                    # only a message without an Action may answer
        break

    print("!! Could not parse an Action. This is the lesson.")
    break

# react_agent.py, Week 2 Lab: the Reasoning Core (ReAct pattern)
# Core idea: make the agent's thinking visible as text:
#   Thought -> Action -> Observation -> Thought -> ... -> Final Answer
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

def get_weather(latitude, longitude, start, end):
    """Same NASA POWER tool as Week 1."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": "T2M,PRECTOTCORR", "community": "ag",
              "latitude": latitude, "longitude": longitude,
              "start": start, "end": end, "format": "JSON"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return json.dumps(r.json()["properties"]["parameter"])

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
        model="deepseek/deepseek-chat",
        messages=messages,
        stop=["Observation:"],                     # the model must not dream one up
    )
    text = response.choices[0].message.content
    print(text.strip(), "\n")
    messages.append({"role": "assistant", "content": text})

    if "Final Answer:" in text:
        break

    match = re.search(
        r'Action:\s*get_weather\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,'
        r'\s*"?(\d{8})"?\s*,\s*"?(\d{8})"?\s*\)', text)
    if not match:
        print("!! Could not parse an Action. This is the lesson.")
        break

    lat, lon, start, end = match.groups()
    obs = get_weather(float(lat), float(lon), start, end)
    print(f"Observation: {obs[:120]}...\n")
    messages.append({"role": "user", "content": f"Observation: {obs}"})

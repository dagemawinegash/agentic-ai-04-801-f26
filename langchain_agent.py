# langchain_agent.py, Week 3: both tools under one framework
# What changes vs. Week 2: the loop, the parser, and the dispatch are gone.
# What does NOT change: our tools. Robust tools survive framework migrations.

import json
import os
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from soil_tool import get_soil          # Week 3 tool, reused as-is

load_dotenv()

# ---- Tools: same logic, now with LangChain's @tool contract -----------------
# The docstring IS the tool description the model reads. Write it like a prompt.

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

@tool
def soil(latitude: float, longitude: float, prop: str = "ph") -> str:
    """Topsoil property (0-20 cm) at a point from iSDAsoil, with uncertainty
    intervals. Falls back to a cached snapshot if the live API is down.
    Useful properties: ph, carbon_organic, nitrogen_total."""
    return get_soil(latitude, longitude, prop)

# ---- The agent: three lines of control flow ---------------------------------

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="deepseek/deepseek-chat-v3-0324",
)

agent = create_agent(
    model=model,
    tools=[weather, soil],
    system_prompt="You are an agronomy assistant for East African "
                  "smallholder contexts. Use your tools; never invent data. "
                  "State data sources and uncertainty in your answer.",
)

# ---- A question that needs BOTH tools ---------------------------------------

question = ("For the location lat -1.94, lon 30.06 near Kigali: what is the "
            "topsoil pH, and how much rain fell in the first week of July "
            "2026? One short paragraph, with sources.")

result = agent.invoke({"messages": [{"role": "user", "content": question}]})

for message in result["messages"]:
    message.pretty_print()

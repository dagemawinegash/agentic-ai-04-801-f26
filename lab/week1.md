# Week 1 Lab: Your First Tool-Using Agent

**Goal:** Build an agent from scratch and watch it make its first autonomous decision. After this lab you can explain, in one sentence and with a running example, what separates an agent from a chatbot.

**A deliberate design choice:** Weeks 1 and 2 use no framework, just the `openai` SDK you already have. The core insight of Week 1 is that *an agent is nothing but a loop in which the model may call tools*. Once you have written that loop yourself, you will understand exactly what frameworks abstract away in Week 3. Start with the framework, and the agent stays magic; start with the loop, and the framework becomes a convenience. (A pragmatic bonus: the raw SDK loop is nearly immune to framework version drift.)

## Module 1.1: The Agent Loop

### Setup

Nothing to install: `requests`, the library the tool uses to call the weather API, came with `requirements.txt` in Week 0. Activate your venv and stay in the repository folder.

### The three parts of every tool-using agent

The script below has exactly three sections, and every agent you build this semester, including your capstone, has the same three:

1. **The tool:** a plain Python function. Here, a real call to NASA POWER, the open agroclimate API we use all semester (keyless, daily data since 1981, 3 to 5 days latency).
2. **The contract:** a JSON schema telling the model what the tool does and what arguments it takes. Read it carefully: the `description` fields are prompts. The model decides whether and how to call your tool based on nothing but this text.
3. **The loop:** call the model; if it requests tool calls, run them and feed the results back; if it answers, stop.

```python
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
    data = r.json()
    label = f"\n[source: NASA POWER, {'/'.join(data['header']['sources'])}]"
    return json.dumps(data["properties"]["parameter"]) + label
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
        model="deepseek/deepseek-chat-v3-0324",
        messages=messages,
        max_tokens=1024,                            # some providers reject calls without a budget
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
```

### Real output (instructor validation run, 27 Sep 2026)

```
[agent calls tool] get_weather({'latitude': -1.94, 'longitude': 30.06, 'start': '20260701', 'end': '20260707'})
A total of 7.03 mm of rain fell in Kigali during the first week of July 2026.
```

**The moment this lab is built around:** nobody told the model to call the tool, which arguments to use, or that "first week of July" means `20260701` to `20260707`. It read the question, read the tool contract, and decided. That is the difference between a chatbot and an agent, demonstrated in about 60 lines.

**Sanity-check the number.** July is dry season in Kigali; about 7 mm over seven days is what reality should deliver. Never trust a tool result you could not sanity-check. If your agent had reported 90 mm, the correct reaction is suspicion, not a screenshot. This habit is graded in the capstone. Your run should say 7.03 mm, the reference value frozen in `weather_snapshot.json` in the repository; if the room sees a different number, tell a TA rather than doubting your setup (see the next paragraph for why that can happen).

**Check the cost side.** Behind the scenes this single script run produced **two** generations, the decision call and the answer call. On the instructor's dashboard their finish reasons read `tool_calls` and `stop`, about USD 0.0003 in total (27 Sep 2026). Run the cost check from Week 0 before and after your own run, and you see your share. Agent token consumption grows with every loop iteration; remember this when your team budgets its capstone.

### Reference values, and why they moved

The same run on 11 Aug 2026 printed **0.9 mm**. Same coordinates, same week, same API, and in September the answer was 7.03 mm. Nothing was wrong on either day. NASA POWER fills its most recent weeks with a provisional near-real-time product (GEOS-IT) and replaces those weeks when the MERRA-2 reanalysis catches up; in August the first week of July was five weeks old and provisional, by late September it was final. The response header says which product you are reading (`"sources": ["GEOSIT"]` for provisional weeks, `["MERRA2"]` for final ones), and in August nobody read the header.

Three consequences that are already in the code above. The tool appends a **source label** to its output (`[source: NASA POWER, MERRA2]`), so the model knows which product it is reading; the label is new since the validation run of 27 Sep: POWER told us in its header that our August values were provisional, the model never saw the header, now it does. The course's **reference values are frozen** in `weather_snapshot.json` (Kigali 7.03 mm, Nairobi 3.85 mm, both MERRA-2, frozen 27 Sep 2026); a test never compares against a live number. And the story becomes **Autopsy 7** in Week 5, where it breaks a sentence this script used to contain: "July 2026 weather will never change, so 0.9 mm cannot rot." It rotted.

### Exercises

Work on a copy, so the original stays intact for the weeks that build on it:

```powershell
copy first_agent.py my_agent.py
```

(macOS or Linux: `cp first_agent.py my_agent.py`.) Then run `python my_agent.py` after each change.

1. Ask about Nairobi instead of Kigali, without giving coordinates. Does the model find them itself? Should you trust coordinates a language model "remembers"?
2. Ask something the tool is useless for ("What is the capital of Rwanda?"). Does the model call the tool anyway? Some models do. What does that tell you about contract quality?
3. Delete the `description` from the tool contract and re-run several times. Watch the call quality degrade. Tool descriptions are prompts; this is the proof.

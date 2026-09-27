# hello_llm.py, Week 0 smoke test
# Goal: one successful LLM call via OpenRouter, with clean key handling.
# If this runs, your environment is ready for Session 1.

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from .env; never hardcode keys

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = client.chat.completions.create(
    model="deepseek/deepseek-chat-v3-0324",   # cheap and capable, our default for labs
    max_tokens=1024,                            # some providers reject calls without a budget
    messages=[
        {"role": "user",
         "content": "In one sentence: what makes an AI system 'agentic'?"}
    ],
)

print(response.choices[0].message.content)
print("---")
print(f"Tokens used: {response.usage.total_tokens} "
      f"(prompt {response.usage.prompt_tokens}, "
      f"completion {response.usage.completion_tokens})")

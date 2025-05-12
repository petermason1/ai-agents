# gemini_test.py

import json
import time
from openai import OpenAI

# —–– CONFIG —––––––––––––––––––––––––––––––––––––––––––––––––––––
GOOGLE_API_KEY = "***REMOVED***"
MODEL_ID        = "gemini-2.5-flash-preview-04-17"
SYSTEM_PROMPT   = "You are a helpful assistant."
USER_QUESTION   = "Give me a summary of today's top tech news."
OUTPUT_FORMAT   = "json"  # "json" or "bullets"
# —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# 1) Instantiate the client pointed at Gemini’s OpenAI-compat endpoint
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 2) Define a retry helper
def ask_with_retries(model, messages, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages
            )
        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {e}")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("All retries failed")

# 3) Build the message payload
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": USER_QUESTION}
]

# 4) Ask Gemini
resp = ask_with_retries(MODEL_ID, messages)
content = resp.choices[0].message.content

# 5) Print nicely
if OUTPUT_FORMAT.lower() == "json":
    print(json.dumps({"answer": content}, indent=2))
else:
    for line in content.splitlines():
        print(f"- {line.strip()}")

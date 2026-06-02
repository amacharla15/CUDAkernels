import requests
import json
import time

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen_lora_merged_v1"

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Explain the difference between fine-tuning and serving in 3 sentences."
        }
    ],
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": False
}

start = time.perf_counter()
response = requests.post(URL, json=payload, timeout=120)
end = time.perf_counter()

print("status_code:", response.status_code)
print("latency_sec:", round(end - start, 4))

if response.status_code != 200:
    print(response.text)
    raise SystemExit(1)

data = response.json()
print(json.dumps(data, indent=2))
print("\nanswer:")
print(data["choices"][0]["message"]["content"])

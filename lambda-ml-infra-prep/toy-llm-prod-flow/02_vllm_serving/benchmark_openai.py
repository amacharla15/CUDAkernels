import json
import time
import statistics
import requests
from pathlib import Path

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "toy-llm-prod-flow" / "02_vllm_serving" / "models" / "qwen_lora_merged_v1"

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen_lora_merged_v1"

PROMPTS = [
    "Explain LoRA forward pass clearly.",
    "What is the difference between fine-tuning and serving?",
    "What is TTFT in LLM inference?",
    "Why does KV cache matter during decode?",
    "Explain batching in an LLM inference server."
]

MAX_TOKENS = 128
REPEAT = 2


def percentile(values, p):
    values = sorted(values)
    index = int((len(values) - 1) * p)
    return values[index]


def count_tokens(tokenizer, text):
    if tokenizer is None:
        return max(1, len(text.split()))
    return len(tokenizer.encode(text, add_special_tokens=False))


def run_one(prompt, tokenizer):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        "stream": True
    }

    start = time.perf_counter()
    first_token_time = None
    output_parts = []

    with requests.post(URL, json=payload, stream=True, timeout=180) as response:
        if response.status_code != 200:
            print(response.text)
            raise RuntimeError(f"Request failed with status {response.status_code}")

        for raw_line in response.iter_lines():
            if not raw_line:
                continue

            line = raw_line.decode("utf-8")

            if not line.startswith("data: "):
                continue

            data = line[len("data: "):]

            if data.strip() == "[DONE]":
                break

            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            token_text = delta.get("content", "")

            if token_text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_parts.append(token_text)

    end = time.perf_counter()

    output_text = "".join(output_parts)
    output_tokens = count_tokens(tokenizer, output_text)

    latency = end - start
    ttft = None
    if first_token_time is not None:
        ttft = first_token_time - start

    tpot = None
    if ttft is not None and output_tokens > 1:
        tpot = (latency - ttft) / (output_tokens - 1)

    tokens_per_sec = output_tokens / latency if latency > 0 else 0.0

    return {
        "prompt": prompt,
        "output_tokens": output_tokens,
        "latency_sec": latency,
        "ttft_sec": ttft,
        "tpot_sec": tpot,
        "tokens_per_sec": tokens_per_sec
    }


def main():
    tokenizer = None

    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

    results = []

    for i in range(REPEAT):
        for prompt in PROMPTS:
            result = run_one(prompt, tokenizer)
            results.append(result)
            print(json.dumps(result, indent=2))

    latencies = [x["latency_sec"] for x in results]
    ttfts = [x["ttft_sec"] for x in results if x["ttft_sec"] is not None]
    tpots = [x["tpot_sec"] for x in results if x["tpot_sec"] is not None]
    throughput = [x["tokens_per_sec"] for x in results]

    summary = {
        "num_requests": len(results),
        "p50_latency_sec": percentile(latencies, 0.50),
        "p90_latency_sec": percentile(latencies, 0.90),
        "p50_ttft_sec": percentile(ttfts, 0.50),
        "p90_ttft_sec": percentile(ttfts, 0.90),
        "avg_tpot_sec": statistics.mean(tpots),
        "avg_tokens_per_sec": statistics.mean(throughput)
    }

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))

    out_dir = ROOT / "toy-llm-prod-flow" / "02_vllm_serving" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    print(f"\nSaved report to: {out_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()

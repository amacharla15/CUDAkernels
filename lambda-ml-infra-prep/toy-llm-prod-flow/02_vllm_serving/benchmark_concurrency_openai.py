import json
import time
import statistics
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "toy-llm-prod-flow" / "02_vllm_serving" / "models" / "qwen_lora_merged_v1"

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen_lora_merged_v1"

PROMPTS = [
    "Explain LoRA forward pass clearly.",
    "What is TTFT in LLM inference?",
    "Why does KV cache matter during decode?",
    "Explain batching in an LLM inference server.",
    "What is the difference between fine-tuning and serving?",
    "Explain why vLLM improves serving throughput.",
    "What is TPOT in LLM inference?",
    "Explain continuous batching simply."
]

CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 16
MAX_TOKENS = 128


def percentile(values, p):
    values = sorted(values)
    index = int((len(values) - 1) * p)
    return values[index]


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

    with requests.post(URL, json=payload, stream=True, timeout=240) as response:
        if response.status_code != 200:
            return {"ok": False, "error": response.text}

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
    output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

    latency = end - start
    ttft = first_token_time - start if first_token_time is not None else None
    tpot = (latency - ttft) / (output_tokens - 1) if ttft is not None and output_tokens > 1 else None
    tokens_per_sec = output_tokens / latency if latency > 0 else 0.0

    return {
        "ok": True,
        "latency_sec": latency,
        "ttft_sec": ttft,
        "tpot_sec": tpot,
        "tokens_per_sec": tokens_per_sec,
        "output_tokens": output_tokens
    }


def run_level(concurrency, tokenizer):
    prompts = []
    for i in range(REQUESTS_PER_LEVEL):
        prompts.append(PROMPTS[i % len(PROMPTS)])

    start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for prompt in prompts:
            futures.append(executor.submit(run_one, prompt, tokenizer))

        for future in as_completed(futures):
            results.append(future.result())

    end = time.perf_counter()

    good = []
    for item in results:
        if item["ok"]:
            good.append(item)

    latencies = [x["latency_sec"] for x in good]
    ttfts = [x["ttft_sec"] for x in good]
    tpots = [x["tpot_sec"] for x in good]
    total_output_tokens = sum(x["output_tokens"] for x in good)
    wall_time = end - start

    summary = {
        "concurrency": concurrency,
        "num_requests": len(results),
        "successful_requests": len(good),
        "wall_time_sec": wall_time,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_sec": total_output_tokens / wall_time,
        "p50_latency_sec": percentile(latencies, 0.50),
        "p90_latency_sec": percentile(latencies, 0.90),
        "p50_ttft_sec": percentile(ttfts, 0.50),
        "p90_ttft_sec": percentile(ttfts, 0.90),
        "avg_tpot_sec": statistics.mean(tpots),
        "avg_per_request_tokens_per_sec": statistics.mean([x["tokens_per_sec"] for x in good])
    }

    return {"summary": summary, "results": results}


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

    all_runs = []

    for concurrency in CONCURRENCY_LEVELS:
        print(f"\nRunning concurrency={concurrency}")
        run = run_level(concurrency, tokenizer)
        all_runs.append(run)
        print(json.dumps(run["summary"], indent=2))

    out_dir = ROOT / "toy-llm-prod-flow" / "02_vllm_serving" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "concurrency_benchmark_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"runs": all_runs}, f, indent=2)

    print(f"\nSaved report to: {out_file}")


if __name__ == "__main__":
    main()
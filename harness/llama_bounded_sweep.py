#!/usr/bin/env python3
"""Bounded mini-sweep for a slow/stalling model.

Runs N sequential requests (concurrency 1) using the exact same
streamed_request function as the article's harness, but enforces a hard
wall-clock cap per request from a watchdog thread. Requests that exceed
the cap are recorded as 'abandoned' with their elapsed time, instead of
blocking the run indefinitely. Each record is appended to the output file
immediately, so no data is lost if the run is interrupted.
"""
import importlib.util
import json
import os
import sys
import threading
import time

spec = importlib.util.spec_from_file_location("bench", "/root/do_latency_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.3-70b-instruct"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
CAP_S = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
OUT = sys.argv[4] if len(sys.argv) > 4 else f"/root/bench_results/bounded_{MODEL}.jsonl"

key = os.environ["DO_MODEL_ACCESS_KEY"]
messages = [{"role": "user", "content": "Reply with a two-sentence summary of why caching matters for LLM inference."}]

for i in range(N):
    result = {}
    start = time.perf_counter()

    def work():
        r = bench.streamed_request("https://inference.do-ai.run/v1", key, MODEL, messages)
        result.update(r)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(CAP_S)
    elapsed = (time.perf_counter() - start) * 1000
    if t.is_alive():
        rec = {"i": i, "abandoned_after_ms": round(elapsed, 1), "ttft_ms": result.get("ttft_ms")}
        print(f"req {i}: ABANDONED after {elapsed:,.0f} ms (ttft={result.get('ttft_ms')})", flush=True)
    else:
        rec = {"i": i, "ttft_ms": result.get("ttft_ms"), "total_ms": result.get("total_ms"),
               "completion_tokens": result.get("completion_tokens"), "error": result.get("error")}
        print(f"req {i}: ttft={result.get('ttft_ms')} total={result.get('total_ms')} err={result.get('error')}", flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")

print("done", flush=True)

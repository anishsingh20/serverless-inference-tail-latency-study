#!/usr/bin/env python3
"""
Latency benchmark for DigitalOcean Serverless Inference.

Measures, per request:
  - TTFT (time to first streamed token), timestamped client-side
  - total completion time
  - prompt and completion token counts, from the usage object

Workloads:
  1. baseline_sweep  - N requests at concurrency 1, 5, and 20
  2. chained_session - M independent chains of `chain_length` sequential
                        calls, each call's output feeding the next call's
                        input, with total chain time recorded per chain

Concurrency is implemented with a real OS-thread pool (ThreadPoolExecutor),
not a single-process asyncio loop. This is a deliberate choice: a 2026
measurement-bias paper (Chandrasekar and Kramberger, arXiv, cited in this
article) shows that single-process asyncio benchmarking clients can hit
their own client-side queueing bottleneck under concurrency, which
inflates the very TTFT numbers you are trying to measure. Real OS threads
release the GIL during blocking network I/O, which avoids that specific
failure mode at the modest concurrency levels used here.

Run:
  export DO_MODEL_ACCESS_KEY=your_key_here
  python3 do_latency_bench.py --model <model-id> --out results_daytime.json
  python3 do_latency_bench.py --model <model-id> --out results_overnight.json
"""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request


def percentile(sorted_vals, pct):
    """Linear-interpolation percentile, no external dependency required."""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize(ttfts_ms):
    s = sorted(t for t in ttfts_ms if t is not None)
    if not s:
        return {"n": 0}
    p50 = percentile(s, 50)
    p95 = percentile(s, 95)
    p99 = percentile(s, 99)
    return {
        "n": len(s),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "p99_to_p50_ratio": round(p99 / p50, 2) if p50 else None,
    }


def streamed_request(base_url, api_key, model, messages, max_tokens=64):
    """Send one streaming chat completion. Return TTFT and total time in ms."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    start = time.perf_counter()
    ttft = None
    content_parts = []
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                chunk = json.loads(body)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                        content_parts.append(delta["content"])
    except Exception as e:
        return {"error": str(e), "ttft_ms": None, "total_ms": None}
    total = (time.perf_counter() - start) * 1000
    return {
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "total_ms": round(total, 1),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "content": "".join(content_parts),
    }


def baseline_sweep(base_url, api_key, model, n_requests, concurrency_levels, prompt):
    results = {}
    messages = [{"role": "user", "content": prompt}]
    for c in concurrency_levels:
        print(f"[baseline] concurrency={c}, {n_requests} requests", flush=True)
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            futures = [
                pool.submit(streamed_request, base_url, api_key, model, messages)
                for _ in range(n_requests)
            ]
            for f in concurrent.futures.as_completed(futures):
                records.append(f.result())
        ttfts = [r.get("ttft_ms") for r in records]
        results[f"concurrency_{c}"] = {
            "summary": summarize(ttfts),
            "records": records,
        }
    return results


def chained_session(base_url, api_key, model, n_chains, chain_length, prompt_prefix):
    chains = []
    for i in range(n_chains):
        history = [{"role": "user", "content": f"{prompt_prefix} Chain {i}, step 0."}]
        start = time.perf_counter()
        calls = []
        for step in range(chain_length):
            r = streamed_request(base_url, api_key, model, history)
            calls.append(r)
            history.append({"role": "assistant", "content": r.get("content") or "ok"})
            history.append(
                {"role": "user", "content": f"Chain {i}, step {step + 1}. Continue."}
            )
        total_chain_ms = (time.perf_counter() - start) * 1000
        chains.append({"chain_index": i, "total_chain_ms": round(total_chain_ms, 1), "calls": calls})
        print(f"[chained] chain {i} done: {total_chain_ms:.0f} ms total", flush=True)
    chain_totals = sorted(c["total_chain_ms"] for c in chains)
    return {
        "chain_summary": {
            "n_chains": len(chain_totals),
            "p50_ms": round(percentile(chain_totals, 50), 1),
            "p95_ms": round(percentile(chain_totals, 95), 1),
            "p99_ms": round(percentile(chain_totals, 99), 1),
        },
        "chains": chains,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://inference.do-ai.run/v1")
    ap.add_argument("--model", required=True, help="Exact model ID from GET /v1/models")
    ap.add_argument("--n-requests", type=int, default=75)
    ap.add_argument("--n-chains", type=int, default=30)
    ap.add_argument("--chain-length", type=int, default=10)
    ap.add_argument("--out", default="latency_bench_results.json")
    args = ap.parse_args()

    api_key = os.environ.get("DO_MODEL_ACCESS_KEY")
    if not api_key:
        raise SystemExit("Set DO_MODEL_ACCESS_KEY before running.")

    results = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print("== Baseline sweep ==", flush=True)
    results["baseline_sweep"] = baseline_sweep(
        args.base_url,
        api_key,
        args.model,
        args.n_requests,
        concurrency_levels=[1, 5, 20],
        prompt="Reply with a two-sentence summary of why caching matters for LLM inference.",
    )

    print("== Chained session ==", flush=True)
    results["chained_session"] = chained_session(
        args.base_url,
        api_key,
        args.model,
        args.n_chains,
        args.chain_length,
        prompt_prefix="You are debugging a production incident.",
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

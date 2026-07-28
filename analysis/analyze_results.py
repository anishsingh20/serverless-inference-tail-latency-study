#!/usr/bin/env python3
"""Analysis and charts for the DO Serverless Inference latency run.

Input:  results_<model>.json files produced by do_latency_bench.py
        (the article's harness) plus bounded_llama3.3-70b-instruct.jsonl
        from the bounded follow-up.
Output: every table in the article, printed as markdown, and the four
        measured charts as PNG files.

Requires: numpy, matplotlib (pip install numpy matplotlib).

Run:  python3 analyze_results.py --data-dir ./bench_results --charts-dir .
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["mistral-3-14B", "openai-gpt-oss-120b", "llama-4-maverick"]
DISPLAY = {"mistral-3-14B": "Mistral 3 14B",
           "openai-gpt-oss-120b": "GPT-OSS 120B",
           "llama-4-maverick": "Llama 4 Maverick"}
COLORS = {"mistral-3-14B": "#0069ff",
          "openai-gpt-oss-120b": "#e85d75",
          "llama-4-maverick": "#2e9e5b"}


def pct(vals, p):
    return float(np.percentile(np.asarray(vals, dtype=float), p, method="linear"))


def clean(records, key):
    return [r[key] for r in records if not r.get("error") and r.get(key) is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--charts-dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.charts_dir, exist_ok=True)

    data = {m: json.load(open(os.path.join(args.data_dir, f"results_{m}.json")))
            for m in MODELS}

    # ---- Table: TTFT by concurrency ----
    print("\n== TTFT by concurrency (ms) ==")
    print("| Model | Concurrency | TTFT p50 | TTFT p95 | TTFT p99 | p99:p50 ratio |")
    print("| --- | --- | --- | --- | --- | --- |")
    for m in MODELS:
        for c in (1, 5, 20):
            t = clean(data[m]["baseline_sweep"][f"concurrency_{c}"]["records"], "ttft_ms")
            print(f"| {DISPLAY[m]} | {c} | {pct(t,50):,.0f} ms | {pct(t,95):,.0f} ms "
                  f"| {pct(t,99):,.0f} ms | {pct(t,99)/pct(t,50):.2f} |")

    # ---- Table: total completion time at concurrency 1 ----
    print("\n== Total completion time, concurrency 1 ==")
    print("| Model | Total p50 | Total p95 | Total p99 | p99:p50 | CV | Median gen speed |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for m in MODELS:
        recs = [r for r in data[m]["baseline_sweep"]["concurrency_1"]["records"] if not r.get("error")]
        t = [r["total_ms"] for r in recs]
        arr = np.asarray(t)
        tps = [r["completion_tokens"] / ((r["total_ms"] - r["ttft_ms"]) / 1000)
               for r in recs if r.get("ttft_ms") and r.get("completion_tokens")]
        print(f"| {DISPLAY[m]} | {pct(t,50):,.0f} ms | {pct(t,95):,.0f} ms | {pct(t,99):,.0f} ms "
              f"| {pct(t,99)/pct(t,50):.2f} | {arr.std(ddof=1)/arr.mean()*100:.0f}% "
              f"| {np.median(tps):.0f} tokens/s |")

    # ---- Table: measured 10-call chains ----
    print("\n== Measured 10-call chains ==")
    print("| Model | Chain p50 | Chain p95 | Worst chain |")
    print("| --- | --- | --- | --- |")
    for m in MODELS:
        tot = [c["total_chain_ms"] / 1000 for c in data[m]["chained_session"]["chains"]]
        print(f"| {DISPLAY[m]} | {pct(tot,50):.1f} s | {pct(tot,95):.1f} s | {max(tot):,.1f} s |")

    # ---- Bounded Llama 3.3 70B follow-up ----
    bpath = os.path.join(args.data_dir, "bounded_llama3.3-70b-instruct.jsonl")
    if os.path.exists(bpath):
        b = [json.loads(l) for l in open(bpath)]
        t = [r["total_ms"] for r in b if r.get("total_ms")]
        f = [r["ttft_ms"] for r in b if r.get("ttft_ms")]
        tps = [r["completion_tokens"] / ((r["total_ms"] - r["ttft_ms"]) / 1000)
               for r in b if r.get("ttft_ms") and r.get("total_ms")]
        print(f"\n== Llama 3.3 70B bounded follow-up: n={len(b)}, "
              f"TTFT p50 {pct(f,50)/1000:.1f} s, total p50 {pct(t,50)/1000:.1f} s, "
              f"total range {min(t)/1000:.0f}-{max(t)/1000:.0f} s, "
              f"median gen speed {np.median(tps):.1f} tokens/s ==")

    # ---- Chart 1: TTFT CDF at concurrency 1 ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in MODELS:
        t = np.sort(clean(data[m]["baseline_sweep"]["concurrency_1"]["records"], "ttft_ms"))
        ax.plot(t, np.arange(1, len(t) + 1) / len(t) * 100,
                label=DISPLAY[m], color=COLORS[m], lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Time to first token (ms, log scale)")
    ax.set_ylabel("Percent of requests at or below")
    ax.set_title("Measured TTFT distribution, concurrency 1, 75 requests per model\n"
                 "DigitalOcean Serverless Inference, July 27, 2026, from nyc2")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.axhline(99, color="gray", ls=":", lw=1)
    ax.text(ax.get_xlim()[0] * 1.05, 51, "p50", fontsize=8, color="gray")
    ax.text(ax.get_xlim()[0] * 1.05, 95.5, "p99", fontsize=8, color="gray")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.charts_dir, "measured-ttft-cdf.png"), dpi=150)

    # ---- Chart 2: TTFT percentiles vs concurrency, one panel per model ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, m in zip(axes, MODELS):
        cs = [1, 5, 20]
        p50s, p95s, p99s = [], [], []
        for c in cs:
            t = clean(data[m]["baseline_sweep"][f"concurrency_{c}"]["records"], "ttft_ms")
            p50s.append(pct(t, 50)); p95s.append(pct(t, 95)); p99s.append(pct(t, 99))
        ax.plot(cs, p50s, "-o", color=COLORS[m], label="p50")
        ax.plot(cs, p95s, "--s", color=COLORS[m], alpha=0.7, label="p95")
        ax.plot(cs, p99s, ":^", color=COLORS[m], alpha=0.5, label="p99")
        ax.set_title(DISPLAY[m], fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_xscale("log"); ax.set_xticks(cs); ax.set_xticklabels(cs)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("TTFT (ms, log scale)")
    fig.suptitle("Measured TTFT p50 / p95 / p99 vs concurrency, 75 requests per cell. "
                 "DigitalOcean Serverless Inference, July 27, 2026", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.charts_dir, "measured-ttft-by-concurrency.png"), dpi=150)

    # ---- Chart 3: task completion vs chain length (bootstrap from measured c=1 totals) ----
    rng = np.random.default_rng(42)
    N_BOOT = 100_000
    chain_lengths = [1, 3, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in MODELS:
        totals = np.asarray([r["total_ms"] for r in
                             data[m]["baseline_sweep"]["concurrency_1"]["records"]
                             if not r.get("error")])
        p50s, p99s = [], []
        for n in chain_lengths:
            sums = rng.choice(totals, size=(N_BOOT, n), replace=True).sum(axis=1)
            p50s.append(np.percentile(sums, 50) / 1000)
            p99s.append(np.percentile(sums, 99) / 1000)
        ax.plot(chain_lengths, p50s, "-o", color=COLORS[m], label=f"{DISPLAY[m]} p50", lw=2, ms=4)
        ax.plot(chain_lengths, p99s, "--", color=COLORS[m], label=f"{DISPLAY[m]} p99", lw=1.2)
        measured = [c["total_chain_ms"] / 1000 for c in data[m]["chained_session"]["chains"]]
        ax.plot([10], [pct(measured, 50)], "*", color=COLORS[m], ms=14, mec="black", mew=0.5)
    ax.set_xlabel("Sequential calls in the agent chain")
    ax.set_ylabel("Task completion time (seconds)")
    ax.set_yscale("log")
    ax.set_title("Task completion time vs chain length, resampled from 75 measured requests per model\n"
                 "Stars: directly measured 30x 10-call chains. DigitalOcean Serverless Inference, July 27, 2026")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.charts_dir, "measured-task-time-vs-chain-length.png"), dpi=150)

    # ---- Chart 4: the metric flip, TTFT winner vs task winner ----
    mA, mB = "llama-4-maverick", "openai-gpt-oss-120b"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ttftA = pct(clean(data[mA]["baseline_sweep"]["concurrency_1"]["records"], "ttft_ms"), 50)
    ttftB = pct(clean(data[mB]["baseline_sweep"]["concurrency_1"]["records"], "ttft_ms"), 50)
    chainA = pct([c["total_chain_ms"] for c in data[mA]["chained_session"]["chains"]], 50) / 1000
    chainB = pct([c["total_chain_ms"] for c in data[mB]["chained_session"]["chains"]], 50) / 1000
    b1 = axes[0].bar([DISPLAY[mA], DISPLAY[mB]], [ttftA, ttftB], color=[COLORS[mA], COLORS[mB]])
    axes[0].bar_label(b1, fmt="%.0f ms", fontsize=11, fontweight="bold")
    axes[0].set_title("Median TTFT, single call\nThe benchmark-headline metric")
    axes[0].set_ylabel("milliseconds")
    axes[0].set_ylim(0, 1400)
    b2 = axes[1].bar([DISPLAY[mA], DISPLAY[mB]], [chainA, chainB], color=[COLORS[mA], COLORS[mB]])
    axes[1].bar_label(b2, fmt="%.1f s", fontsize=11, fontweight="bold")
    axes[1].set_title("Median measured 10-call agent chain\nThe metric an agent user waits on")
    axes[1].set_ylabel("seconds")
    axes[1].set_ylim(0, 30)
    for a in axes:
        a.grid(alpha=0.3, axis="y")
    fig.suptitle("Same two models, opposite winners. Measured on DigitalOcean Serverless Inference, "
                 "July 27, 2026 (75 single calls and 30 chains per model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.charts_dir, "measured-ttft-winner-vs-task-winner.png"), dpi=150)

    print(f"\nCharts written to {args.charts_dir}")


if __name__ == "__main__":
    main()

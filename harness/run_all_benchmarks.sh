#!/bin/bash
# Orchestration script for the July 27, 2026 run.
# Ran from /root on a DigitalOcean GPU Droplet (gpu-h200x1-141gb, NYC2).
# /root/.do_bench_env contains one line: export DO_MODEL_ACCESS_KEY=...
source /root/.do_bench_env
cd /root
mkdir -p bench_results
echo "=== RUN STARTED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
for M in mistral-3-14B openai-gpt-oss-120b llama-4-maverick; do
  echo "=== MODEL: $M start $(date -u +%H:%M:%SZ) ==="
  python3 do_latency_bench.py --model "$M" --out "bench_results/results_${M}.json"
  echo "=== MODEL: $M done $(date -u +%H:%M:%SZ) ==="
done
echo "=== MODEL: llama3.3-70b-instruct (sweep + 1 chain) start $(date -u +%H:%M:%SZ) ==="
python3 do_latency_bench.py --model llama3.3-70b-instruct --n-chains 1 --out bench_results/results_llama3.3-70b-instruct.json
echo "=== MODEL: llama3.3-70b-instruct done $(date -u +%H:%M:%SZ) ==="
echo "=== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

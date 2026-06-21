#!/usr/bin/env bash
# E1 — the 2x4 VEST grid.
#
# Runs the probe on each (model, dataset) cell sequentially on a single
# GPU and writes one parquet per cell. After all cells finish, runs
# `pear decompose` per cell and assembles the audit table.
#
# Total wall time on one H100 80 GB (bf16, no flash-attn): ~5 hours.
# Resumable: cells that already have an output parquet are skipped.
#
# Usage:
#   bash scripts/run_e1_grid.sh
#   bash scripts/run_e1_grid.sh --n 200          # smaller per-cell sample
#
set -uo pipefail

cd "$(dirname "$0")/.."
N_PER_SOURCE=800
N_REALWORLD=600
if [[ "${1:-}" == "--n" ]]; then
    N_PER_SOURCE="${2:?--n requires a value}"
    N_REALWORLD="$N_PER_SOURCE"
fi

LOG=results/audits/e1_grid.log
mkdir -p results/probes results/vest results/audits
echo "[grid $(date -u +%H:%M:%S)] starting E1 grid (N=$N_PER_SOURCE)" | tee "$LOG"

run_probe () {
    local model_id="$1"; local source="$2"; local tag="$3"; local n="$4"
    local out="results/probes/probe_${tag}.parquet"
    local pl="results/probes/probe_${tag}.log"
    if [[ -f "$out" ]]; then
        echo "[grid $(date -u +%H:%M:%S)] SKIP $tag (parquet exists)" | tee -a "$LOG"
        return 0
    fi
    echo "[grid $(date -u +%H:%M:%S)] START $tag" | tee -a "$LOG"
    python -m pear probe \
        --model-id "$model_id" --source "$source" --n-per-source "$n" \
        --out "$out" --seed 0 > "$pl" 2>&1
    echo "[grid $(date -u +%H:%M:%S)] END   $tag (exit $?)" | tee -a "$LOG"
}

# 9B cells first (largest model loaded once, all four datasets, then 2B).
run_probe "Qwen/Qwen3.5-9B" "chartqa"     "qwen35_9b_chartqa"     "$N_PER_SOURCE"
run_probe "Qwen/Qwen3.5-9B" "ai2d"        "qwen35_9b_ai2d"        "$N_PER_SOURCE"
run_probe "Qwen/Qwen3.5-9B" "textvqa"     "qwen35_9b_textvqa"     "$N_PER_SOURCE"
run_probe "Qwen/Qwen3.5-9B" "realworldqa" "qwen35_9b_realworldqa" "$N_REALWORLD"

run_probe "Qwen/Qwen3.5-2B" "chartqa"     "qwen35_2b_chartqa"     "$N_PER_SOURCE"
run_probe "Qwen/Qwen3.5-2B" "ai2d"        "qwen35_2b_ai2d"        "$N_PER_SOURCE"
run_probe "Qwen/Qwen3.5-2B" "textvqa"     "qwen35_2b_textvqa"     "$N_PER_SOURCE"
run_probe "Qwen/Qwen3.5-2B" "realworldqa" "qwen35_2b_realworldqa" "$N_REALWORLD"

# Per-cell VEST.
for tag in qwen35_9b_chartqa qwen35_9b_ai2d qwen35_9b_textvqa qwen35_9b_realworldqa \
           qwen35_2b_chartqa qwen35_2b_ai2d qwen35_2b_textvqa qwen35_2b_realworldqa; do
    if [[ -f "results/probes/probe_${tag}.parquet" ]]; then
        python -m pear decompose \
            --parquet "results/probes/probe_${tag}.parquet" \
            > "results/vest/vest_${tag}.txt" 2>&1
    fi
done

# Roll up into a single audit table.
python -m pear audit \
    --entry 9B_chartqa:results/probes/probe_qwen35_9b_chartqa.parquet \
    --entry 9B_ai2d:results/probes/probe_qwen35_9b_ai2d.parquet \
    --entry 9B_textvqa:results/probes/probe_qwen35_9b_textvqa.parquet \
    --entry 9B_realworldqa:results/probes/probe_qwen35_9b_realworldqa.parquet \
    --entry 2B_chartqa:results/probes/probe_qwen35_2b_chartqa.parquet \
    --entry 2B_ai2d:results/probes/probe_qwen35_2b_ai2d.parquet \
    --entry 2B_textvqa:results/probes/probe_qwen35_2b_textvqa.parquet \
    --entry 2B_realworldqa:results/probes/probe_qwen35_2b_realworldqa.parquet \
    --out results/audits/audit_e1_full_grid.csv \
    > results/audits/audit_e1_full_grid.txt 2>&1

echo "[grid $(date -u +%H:%M:%S)] DONE  (see results/audits/audit_e1_full_grid.{csv,txt})" | tee -a "$LOG"

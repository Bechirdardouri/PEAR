#!/usr/bin/env bash
# E3 — base vs. perception-aware checkpoint contrast.
#
# Of the ten perception-aware RL methods surveyed in REPORT.md (PAPO,
# VPPO, PGPO, PRPO, Vision-SR1, SRPO, PDCR, PEPO, VGPO, Perceval),
# only VGPO has released model weights publicly. We probe both the
# base Qwen2.5-VL-7B-Instruct and MuMing0102/VGPO-RL-7B on ChartQA.
#
# Wall time on one H100 80 GB: ~2 hours (n=400 per side).
#
set -uo pipefail
cd "$(dirname "$0")/.."
N=400
LOG=results/audits/e3_vgpo.log
mkdir -p results/probes results/vest results/audits
echo "[e3 $(date -u +%H:%M:%S)] starting E3 VGPO contrast (N=$N per side)" | tee "$LOG"

run_probe () {
    local model_id="$1"; local tag="$2"
    local out="results/probes/probe_${tag}.parquet"
    local pl="results/probes/probe_${tag}.log"
    if [[ -f "$out" ]]; then
        echo "[e3] SKIP $tag (parquet exists)" | tee -a "$LOG"
        return 0
    fi
    echo "[e3 $(date -u +%H:%M:%S)] START $tag" | tee -a "$LOG"
    python -m pear probe \
        --model-id "$model_id" --source chartqa --n-per-source "$N" \
        --out "$out" --seed 0 > "$pl" 2>&1
    echo "[e3 $(date -u +%H:%M:%S)] END   $tag (exit $?)" | tee -a "$LOG"
}

run_probe "Qwen/Qwen2.5-VL-7B-Instruct" "qwen25vl_7b_base_chartqa"
run_probe "MuMing0102/VGPO-RL-7B"        "qwen25vl_7b_vgpo_chartqa"

for tag in qwen25vl_7b_base_chartqa qwen25vl_7b_vgpo_chartqa; do
    python -m pear decompose \
        --parquet "results/probes/probe_${tag}.parquet" \
        > "results/vest/vest_${tag}.txt" 2>&1
done

python -m pear audit \
    --entry base_Qwen25VL7B:results/probes/probe_qwen25vl_7b_base_chartqa.parquet \
    --entry vgpo_Qwen25VL7B:results/probes/probe_qwen25vl_7b_vgpo_chartqa.parquet \
    --out results/audits/audit_e3_vgpo_chartqa.csv \
    > results/audits/audit_e3_vgpo_chartqa.txt 2>&1

echo "[e3 $(date -u +%H:%M:%S)] DONE  (see results/audits/audit_e3_vgpo_chartqa.{csv,txt})" | tee -a "$LOG"

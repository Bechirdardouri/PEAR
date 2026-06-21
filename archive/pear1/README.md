# PEAR — Perceptual Edge-of-competence And Reachability

A 1–2 GPU-day go/no-go experiment for the **PEAR** hypothesis.

## The one falsifiable claim

> For a VLM, the perceptual response curve (how its confidence in the
> gold answer collapses as visual tokens are noised) carries
> information about **RL-learnability** that **difficulty alone does
> not** — specifically, among examples the model currently fails
> (`pass@1 == 0`), the curve's clean-image support `m0` separates
> *reachable* (`pass@k > 0` → RL can get gradient) from *unreadable*
> (`pass@k == 0` → RL is dead), **after controlling for difficulty**.

Everything downstream — curriculum, abstention — is contingent on this
claim. If it fails, PEAR is no-go and you've saved months. If it
holds, you have a green light for the full method.

## Decision rule (printed by `analyze`)

| AUROC(`m0`→reachable \| hard)      | ΔAUROC(PEAR vs difficulty)         | Verdict   |
| ---------------------------------- | ---------------------------------- | --------- |
| ≥ 0.60 (mean over deciles)         | ≥ 0.03, bootstrap 95 % CI excl. 0  | **GO**    |
| 0.55–0.60 *or* ΔAUROC 0.01–0.03    | —                                  | ITERATE   |
| ≈ 0.50                             | ≈ 0                                | **NO-GO** |

NO-GO means the curve is difficulty in disguise → pivot.

## Repo map

| File                       | Purpose                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| [pear/config.py](pear/config.py)         | `Config` dataclass — every knob in one place.                                  |
| [pear/model.py](pear/model.py)           | Load model/processor, `find_vision_module`, `NoiseHook` (additive Gaussian on vision-module output). |
| [pear/scoring.py](pear/scoring.py)       | `teacher_forced_logprob` (gold-answer NLL), `sample_answers`, Qwen3.5 chat builder. |
| [pear/verifiers.py](pear/verifiers.py)   | `verify(pred, gold, ans_type)` for mc / numeric / exact / anls.                |
| [pear/data.py](pear/data.py)             | ChartQA, AI2D, TextVQA loaders + `build_probe_set` + synthetic smoke set.      |
| [pear/probe.py](pear/probe.py)           | `response_curve`, `fit_curve` → `m0, sigma_star, amplitude`.                   |
| [pear/difficulty.py](pear/difficulty.py) | `evaluate_sampling` — G samples with image + G with blank baseline.            |
| [pear/run.py](pear/run.py)               | Orchestrator; streaming parquet writer with resume.                            |
| [pear/analysis.py](pear/analysis.py)     | H1 (decisive) + H2 (NEED axis) + figures + verdict.                            |
| [pear/smoke.py](pear/smoke.py)           | End-to-end pipeline on 8 synthetic examples (<2 min).                          |
| [pear/__main__.py](pear/__main__.py)     | CLI dispatch.                                                                  |

## Setup

### Scaleway H100 (recommended — robust, pinned, idempotent)

```bash
bash pear/setup_h100.sh
source .venv/bin/activate
python -m pear doctor          # full env diagnostic; exits non-zero on any FAIL
```

The script (idempotent, safe to re-run) installs into `./.venv`:

| Component             | Pinned version             | Why                              |
| --------------------- | -------------------------- | -------------------------------- |
| PyTorch               | 2.7.0 + cu126              | Hopper SM 9.0 first-class       |
| flash-attention 2     | 2.7.4.post1 (prebuilt)     | ~1.3-2x speedup on H100          |
| transformers          | 4.62.1                     | First stable Qwen3.5 release    |
| numpy                 | 2.1.3 (capped < 3)         | No surprise breaking changes    |
| protobuf              | 5.29.3                     | Tokenizers compatibility        |
| All others            | See pear/requirements.txt + pear/constraints.txt |

Uses `uv` (fast resolver, installed by the script) with pinned
requirements + transitive constraints — no dependency drift.

Flags:
- `bash pear/setup_h100.sh --skip-torch`  — leave existing torch alone
- `bash pear/setup_h100.sh --verify`      — re-run `pear doctor` only

`HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `TOKENIZERS_PARALLELISM`, and
`PYTHONUNBUFFERED` are exported automatically when you activate the
venv (written to `.venv/bin/activate.d_pear.sh`).

### Generic / non-H100

```bash
pip install --constraint pear/constraints.txt -r pear/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124    # or your CUDA
```

## Run recipe

```bash
# 0. Bootstrap (only once on a fresh H100).
bash pear/setup_h100.sh

# 1. Activate + diagnose. Doctor exits non-zero on any FAIL.
source .venv/bin/activate
python -m pear doctor

# 2. Confirm we can locate the vision module on Qwen3.5-2B (~30s).
python -m pear print-modules

# 3. Smoke test: full pipeline on 8 synthetic examples (~2 min).
python -m pear smoke

# 4. Build the probe set (~2400 examples by default; cached).
python -m pear build

# 5. Run the probe + sampling pipeline. Streams to parquet; resumable.
python -m pear probe

# 6. Analyse, print GO / ITERATE / NO-GO, write 3 figures.
python -m pear analyze
```

Override config from the CLI: `python -m pear probe --model-id Qwen/Qwen3.5-2B --G 12 --K 5`.

## Expected outputs

- `outputs/results.parquet` — one row per example with columns:
  `id, source, answer_type, m0, m_inf, amplitude, sigma_star,
   margins (list[float], length K), pass_at_1, pass_at_k, pass_rate,
   mean_logprob, blank_pass_rate, blank_pass_at_k`.
- `outputs/figs/scatter_difficulty_vs_sigma_star.png` — hard subset,
  colored by reachability. The decisive 2-axis dissociation plot.
- `outputs/figs/curves_by_regime.png` — 9 example response curves
  (3 reachable-hard, 3 unreadable-hard, 3 easy).
- `outputs/figs/auroc_per_bin.png` — within-decile AUROC of `m0`.

## Extension hooks

- **Add a dataset:** implement `load_X(n)` in `pear/data.py` returning
  records `{id, image (PIL), question, answer, answer_type, source}`
  and register it in `build_probe_set`.
- **Try a different VLM:** `python -m pear probe --model-id <id>`.
  The vision-module detector handles `visual` / `vision_model` /
  `vision_tower` automatically; pass `--vision-module foo.bar.baz`
  to override.
- **Different noise type:** edit `NoiseHook.__call__` in `pear/model.py`.

## Out of scope (deliberately)

No RL training, no GRPO/DAPO wiring, no abstention head, no
curriculum scheduler, no multi-GPU. Those are downstream of this
go/no-go and should be built only after a green verdict.

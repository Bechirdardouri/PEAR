"""Orchestration: probe + difficulty per example → parquet (resumable)."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .config import Config
from .data import Record, build_probe_set
from .difficulty import evaluate_sampling
from .model import find_vision_module, load_model_and_processor
from .probe import fit_curve, response_curve


# Parquet schema. `margins` is variable length but the K is fixed by cfg.
def _schema() -> pa.Schema:
    return pa.schema([
        ("id", pa.string()),
        ("source", pa.string()),
        ("answer_type", pa.string()),
        ("m0", pa.float64()),
        ("m_inf", pa.float64()),
        ("amplitude", pa.float64()),
        ("sigma_star", pa.float64()),
        ("margins", pa.list_(pa.float64())),
        ("pass_at_1", pa.bool_()),
        ("pass_at_k", pa.bool_()),
        ("pass_rate", pa.float64()),
        ("mean_logprob", pa.float64()),
        ("blank_pass_at_k", pa.bool_()),
        ("blank_pass_rate", pa.float64()),
    ])


def _already_done(parquet_path: Path) -> set[str]:
    if not parquet_path.exists():
        return set()
    try:
        t = pq.read_table(parquet_path, columns=["id"])
        return set(t.column("id").to_pylist())
    except Exception:
        return set()


def _process_one(model, processor, vision_module, record: Record, cfg: Config,
                 example_seed: int) -> dict:
    margins = response_curve(model, processor, vision_module, record, cfg, example_seed)
    fit = fit_curve(margins, cfg.sigmas)
    diff = evaluate_sampling(model, processor, record, cfg, example_seed)
    return {
        "id": record["id"],
        "source": record["source"],
        "answer_type": record["answer_type"],
        "m0": fit.m0,
        "m_inf": fit.m_inf,
        "amplitude": fit.amplitude,
        "sigma_star": fit.sigma_star,
        "margins": list(map(float, margins.tolist())),
        "pass_at_1": diff.pass_at_1,
        "pass_at_k": diff.pass_at_k,
        "pass_rate": diff.pass_rate,
        "mean_logprob": diff.mean_logprob,
        "blank_pass_at_k": diff.blank_pass_at_k,
        "blank_pass_rate": diff.blank_pass_rate,
    }


def run_pipeline(cfg: Config, records: Iterable[Record] | None = None) -> Path:
    """Main entry. If ``records`` is None, builds from cfg via ``build_probe_set``.

    Streams to ``cfg.parquet_path`` in chunks of 100 rows; resumable.
    Returns the parquet path.
    """
    cfg.ensure_dirs()
    if records is None:
        records = build_probe_set(cfg)
    records = list(records)

    done = _already_done(cfg.parquet_path)
    if done:
        print(f"[run] resume: {len(done)} rows already in {cfg.parquet_path}")
    todo = [r for r in records if r["id"] not in done]
    print(f"[run] processing {len(todo)} / {len(records)} examples")

    print(f"[run] loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)
    vision_module = find_vision_module(model, cfg.vision_module_name)

    schema = _schema()
    writer = pq.ParquetWriter(str(cfg.parquet_path), schema)
    # If resuming, the existing file is opened with a new writer — that
    # creates a separate file. We instead append by writing to a side
    # file and concatenating, OR (simpler) re-load existing rows and
    # write a fresh file. We choose the latter for simplicity.
    if done:
        existing = pq.read_table(cfg.parquet_path, schema=schema)
        writer.write_table(existing)

    buffer: list[dict] = []
    BUFFER_SIZE = 100
    t0 = time.time()
    try:
        for i, record in enumerate(tqdm(todo, desc="pear")):
            try:
                row = _process_one(model, processor, vision_module, record, cfg,
                                   example_seed=hash(record["id"]) % (2**31))
            except Exception as e:
                print(f"[run] FAILED id={record['id']}: {e}")
                traceback.print_exc()
                continue
            buffer.append(row)
            if len(buffer) >= BUFFER_SIZE:
                _flush(writer, buffer, schema)
                buffer.clear()
        if buffer:
            _flush(writer, buffer, schema)
    finally:
        writer.close()

    dt = time.time() - t0
    print(f"[run] done in {dt/60:.1f} min → {cfg.parquet_path}")
    return cfg.parquet_path


def _flush(writer, buffer: list[dict], schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(buffer, schema=schema)
    writer.write_table(table)

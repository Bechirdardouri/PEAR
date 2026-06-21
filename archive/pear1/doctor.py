"""`python -m pear doctor` — environment diagnostic.

Validates every component PEAR depends on, prints a clean report,
and exits with non-zero on any FAIL. Designed to be the single
"is my box ready?" check after running setup_h100.sh.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from typing import Callable

GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}OK  {RESET}{msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET} {msg}")


def _section(title: str) -> None:
    print(f"\n{DIM}— {title} —{RESET}")


def _check_import(name: str, label: str | None = None) -> tuple[bool, str]:
    """Try to import a module; return (ok, version_or_error)."""
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        return True, ver
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_torch() -> int:
    _section("PyTorch / CUDA")
    fails = 0
    ok, info = _check_import("torch")
    if not ok:
        _fail(f"torch: {info}")
        return 1
    import torch
    _ok(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() = False (driver/wheel mismatch?)")
        return 1
    _ok(f"CUDA runtime version: {torch.version.cuda}")
    n_dev = torch.cuda.device_count()
    _ok(f"visible GPUs: {n_dev}")
    for i in range(n_dev):
        cc = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        msg = f"  [{i}] {name}  SM {cc[0]}.{cc[1]}  {mem_gb:.1f} GB"
        if cc[0] >= 9:
            _ok(msg + "  (Hopper or newer)")
        elif cc[0] == 8:
            _ok(msg + "  (Ampere)")
        else:
            _warn(msg + "  (older; bf16 may be slow)")
    # Real op
    try:
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        y = (x @ x).sum().item()
        _ok(f"bfloat16 GPU matmul OK (sum={y:.2f})")
    except Exception as e:
        _fail(f"bf16 GPU op failed: {e}")
        fails += 1
    return fails


def _check_attention() -> int:
    _section("Attention backends")
    fails = 0
    try:
        import torch
        # SDPA is always present in torch >= 2.0
        _ok("torch.nn.functional.scaled_dot_product_attention available")
    except Exception:
        pass
    ok, info = _check_import("flash_attn")
    if ok:
        _ok(f"flash-attention 2 installed: {info}")
        # FA2 specifically needs SM 8.0+; FA3 needs SM 9.0
        try:
            import torch
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 9:
                _ok("  SM 9.0 detected — FA2 will use Hopper kernels")
            elif cc[0] == 8:
                _ok("  SM 8.x detected — FA2 will use Ampere kernels")
        except Exception:
            pass
    else:
        _warn("flash-attention not installed — PEAR will fall back to SDPA (works, ~1.3-2x slower)")
    return fails


def _check_pkg_versions() -> int:
    """Check that every pinned package imports at the expected version."""
    _section("Pinned packages")
    fails = 0
    expected = {
        "transformers": "5.",
        "accelerate": "1.",
        "datasets": "5.",
        "tokenizers": "0.2",
        "safetensors": "0.",
        "huggingface_hub": "1.",
        "PIL": None,   # pillow imports as PIL
        "qwen_vl_utils": "0.0",
        "sentencepiece": None,
        "google.protobuf": None,
        "einops": "0.8",
        "numpy": "2",
        "pandas": "3.",
        "pyarrow": "24",
        "sklearn": "1.",
        "matplotlib": "3.",
        "tqdm": "4.",
    }
    for mod, prefix in expected.items():
        ok, ver = _check_import(mod)
        if not ok:
            _fail(f"{mod}: {ver}")
            fails += 1
            continue
        if prefix is None or ver.startswith(prefix):
            _ok(f"{mod} {ver}")
        else:
            _warn(f"{mod} {ver}  (expected {prefix}.*; may still work)")
    return fails


def _check_pear() -> int:
    _section("PEAR package")
    fails = 0
    for mod in ("pear.config", "pear.model", "pear.scoring",
                "pear.verifiers", "pear.data", "pear.probe",
                "pear.difficulty", "pear.run", "pear.analysis"):
        ok, info = _check_import(mod)
        if ok:
            _ok(mod)
        else:
            _fail(f"{mod}: {info}")
            fails += 1
    return fails


def _check_hf_cache() -> int:
    _section("HuggingFace cache")
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    if os.path.isdir(home) and os.access(home, os.W_OK):
        _ok(f"HF_HOME = {home}  (writable)")
    else:
        _warn(f"HF_HOME = {home}  (missing or read-only; downloads will fail)")
    hf_transfer_ok, _ = _check_import("hf_transfer")
    if os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1":
        _ok("HF_XET_HIGH_PERFORMANCE=1 (fast Xet downloads)")
    elif hf_transfer_ok and os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        _ok("hf_transfer enabled (legacy fast downloads)")
    else:
        _warn("Neither HF_XET_HIGH_PERFORMANCE nor hf_transfer enabled (downloads slower)")
    return 0


def _check_host() -> int:
    _section("Host")
    print(f"  python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  platform : {platform.platform()}")
    return 0


def doctor() -> int:
    print("PEAR environment doctor")
    fails = 0
    fails += _check_host()
    fails += _check_torch()
    fails += _check_attention()
    fails += _check_pkg_versions()
    fails += _check_pear()
    fails += _check_hf_cache()
    print()
    if fails == 0:
        print(f"{GREEN}== ALL CHECKS PASSED =={RESET}")
        return 0
    print(f"{RED}== {fails} CHECK(S) FAILED =={RESET}")
    return 1


def main() -> int:
    return doctor()


if __name__ == "__main__":
    raise SystemExit(main())

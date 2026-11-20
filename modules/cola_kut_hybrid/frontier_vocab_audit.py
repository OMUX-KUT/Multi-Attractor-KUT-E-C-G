"""Run 32k-256k synthetic scale audit + real o200k_base (and optional Gemma 256k) bit-packed L3 audit."""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BIN_PATH = Path("/tmp/sparse_vocab_l3_bench")
STRUCT_SET = {"def", "return", "(", ")", ":", ",", "->", "\n", "    "}
TYPE_SET = {"int", "float", "bool"}
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUM_RE = re.compile(r"^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$")
OP_SET = {"+", "-", "*", "/", "//", "%", "<=", ">=", "==", "!=", "and", "or", "not", "(", ")", ","}
SAFE_BUILTINS = {"abs", "min", "max"}


def classify_token_roles(raw_str: str) -> List[int]:
    """Map decoded subword string into the 5 KUT-110D Layer-1 token roles."""
    s = raw_str.replace("▁", " ").strip()
    roles: List[int] = []
    if not s:
        return roles
    if s in STRUCT_SET:
        roles.append(0)  # STRUCT
    if IDENT_RE.match(s) and s not in TYPE_SET and s not in ("def", "return", "and", "or", "not", "True", "False"):
        roles.append(1)  # IDENT_DECL
        roles.append(4)  # SCOPE_IDENT candidate pool
    if s in TYPE_SET:
        roles.append(2)  # TYPE_NAME
    if IDENT_RE.match(s) or NUM_RE.match(s) or s in OP_SET or s in SAFE_BUILTINS or s in ("True", "False"):
        roles.append(3)  # EXPR_ATOM
    if s in SAFE_BUILTINS:
        roles.append(4)  # SCOPE_IDENT
    return roles


def write_bitmask_bin(token_strings: List[str], vocab_size: int, out_path: Path) -> Dict[str, int]:
    words_per_role = (vocab_size + 63) // 64
    masks = [[0] * words_per_role for _ in range(5)]
    counts = [0] * 5
    for tid in range(min(vocab_size, len(token_strings))):
        for r in classify_token_roles(token_strings[tid]):
            masks[r][tid >> 6] |= (1 << (tid & 63))
            counts[r] += 1
    with out_path.open("wb") as f:
        for r in range(5):
            for w in masks[r]:
                f.write(struct.pack("<Q", w))
    return {f"role_{r}_active_tokens": counts[r] for r in range(5)}


def run_real_mask_cpp(label: str, vocab_size: int, bin_file: Path) -> List[Dict[str, object]]:
    out = subprocess.check_output(
        [str(BIN_PATH), "--real-mask", f"{label}:{vocab_size}", str(bin_file)],
        text=True,
    )
    return json.loads(out)


def try_real_o200k() -> Tuple[str, List[Dict[str, object]], Dict[str, int]]:
    try:
        import tiktoken  # type: ignore
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=False)
        try:
            import tiktoken  # type: ignore
        except ImportError:
            return "SKIPPED_TIKTOKEN_UNAVAILABLE", [], {}

    enc = tiktoken.get_encoding("o200k_base")
    vsize = enc.n_vocab
    tok_strs: List[str] = []
    for i in range(vsize):
        try:
            tok_strs.append(enc.decode_single_token_bytes(i).decode("utf-8", errors="ignore"))
        except Exception:
            tok_strs.append("")
    mask_path = Path("/tmp/o200k_real_mask.bin")
    role_counts = write_bitmask_bin(tok_strs, vsize, mask_path)
    rows = run_real_mask_cpp("o200k_base_real", vsize, mask_path)
    return "MEASURED_REAL_O200K_BASE", rows, role_counts


def try_real_gemma_256k() -> Tuple[str, List[Dict[str, object]], Dict[str, int]]:
    """Load official Gemma 256k vocab ONLY if local file exists or HF_TOKEN is explicitly provided."""
    local_candidates = [
        Path("/content/gemma_tokenizer.json"),
        Path("/content/tokenizer.json"),
        Path("/content/tokenizer.model"),
    ]
    tok_strs: Optional[List[str]] = None
    src_note = ""

    for p in local_candidates:
        if p.exists() and p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            vocab_map = data.get("model", {}).get("vocab", {})
            if isinstance(vocab_map, dict) and len(vocab_map) >= 250000:
                vsize = max(vocab_map.values()) + 1
                tok_strs = [""] * vsize
                for k, idx in vocab_map.items():
                    tok_strs[int(idx)] = str(k)
                src_note = f"MEASURED_LOCAL_GEMMA_JSON:{p.name}"
                break

    if tok_strs is None and os.environ.get("HF_TOKEN"):
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=os.environ["HF_TOKEN"])
            vsize = len(tok)
            tok_strs = [tok.convert_ids_to_tokens(i) or "" for i in range(vsize)]
            src_note = "MEASURED_HF_OFFICIAL_GEMMA_256K"
        except Exception as exc:
            return f"SKIPPED_GEMMA_HF_ERROR:{type(exc).__name__}", [], {}

    if tok_strs is None:
        return "SKIPPED_NEEDS_HF_TOKEN_OR_LOCAL_GEMMA_TOKENIZER", [], {}

    vsize = len(tok_strs)
    mask_path = Path("/tmp/gemma256k_real_mask.bin")
    role_counts = write_bitmask_bin(tok_strs, vsize, mask_path)
    rows = run_real_mask_cpp("gemma_256k_real", vsize, mask_path)
    return src_note, rows, role_counts


def main() -> None:
    synth_rows = json.loads(subprocess.check_output([str(BIN_PATH)], text=True))
    o200k_status, o200k_rows, o200k_counts = try_real_o200k()
    gemma_status, gemma_rows, gemma_counts = try_real_gemma_256k()

    all_rows = synth_rows + o200k_rows + gemma_rows
    all_pass = all(bool(r["phase2a_pass"]) for r in all_rows)

    payload = {
        "all_measured_gates_pass": all_pass,
        "o200k_real_status": o200k_status,
        "o200k_real_role_counts": o200k_counts,
        "gemma_256k_real_status": gemma_status,
        "gemma_256k_real_role_counts": gemma_counts,
        "rows": all_rows,
    }
    Path("logs/phase2a_frontier_vocab_l3.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== [FRONTIER VOCAB L3 BENCHMARK SUMMARY (32k -> 256k)] ===")
    print(f"O200K_REAL_STATUS = {o200k_status}")
    print(f"GEMMA_REAL_STATUS = {gemma_status}")
    for r in all_rows:
        print(
            f"[{r['vocab_label']:24s}][{r['mode']:23s}] "
            f"V={r['vocab_size']:7d} | mask={r['bitpacked_mask_bytes']:7,d}B | "
            f"L1={r['total_layer1_bytes']:9,d}B ({r['l3_utilization_ratio']*100:5.2f}%) | "
            f"density={r['active_bit_density']*100:5.2f}% | TPS={r['measured_tps']:10,.1f} | PASS={r['phase2a_pass']}"
        )
    print(f"ALL_FRONTIER_GATES_PASS={all_pass}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()

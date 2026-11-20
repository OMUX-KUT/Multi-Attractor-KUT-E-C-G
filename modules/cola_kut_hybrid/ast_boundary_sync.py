"""Reversible Python lexer, AST verifier, and Mod0/Mod1/AST-Sync sequence builder."""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass
from typing import List, Literal

ConditionType = Literal["Mod0", "Mod1", "AST-Sync"]

_REVERSIBLE_TOKEN_RE = re.compile(
    r"\s+|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|==|!=|<=|>=|->|\S"
)

_STRUCTURAL_SYMBOLS = {
    "(", ")", "[", "]", "{", "}", ":", ",", ".", "=", "+", "-", "*", "/",
    "==", "!=", "<=", ">=", "->", "<", ">",
}


@dataclass(frozen=True)
class SyncedSequence:
    condition: ConditionType
    raw_tokens: List[str]
    model_tokens: List[str]
    structural_mask: List[bool]
    patch_size: int
    block_size: int
    leading_shift: int
    trailing_pad: int
    is_mod0_aligned: bool
    source_ast_valid: bool


def reversible_tokenize(source: str) -> List[str]:
    """Losslessly split Python source so ''.join(tokens) == source."""
    tokens = _REVERSIBLE_TOKEN_RE.findall(source)
    if "".join(tokens) != source:
        raise AssertionError("Reversible lexer invariant violated")
    return tokens


def is_structural_ast_token(tok: str) -> bool:
    """Identify tokens that form the AST structural skeleton (keywords, whitespace, ops)."""
    if tok.isspace():
        return True
    if keyword.iskeyword(tok):
        return True
    if tok in _STRUCTURAL_SYMBOLS:
        return True
    return False


def verify_ast_integrity(source: str) -> bool:
    """Return True iff the decoded Python string parses with zero SyntaxError."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def compute_ast_sync_pad(length: int, patch_size: int) -> int:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    return (patch_size - (length % patch_size)) % patch_size


def build_three_conditions(
    source: str,
    patch_size: int = 2,
    block_size: int = 16,
) -> List[SyncedSequence]:
    """Build Mod0, Mod1 (real 1-token boundary shift), and AST-Sync sequences."""
    ast_ok = verify_ast_integrity(source)
    base = reversible_tokenize(source)
    if not base:
        raise ValueError("Empty token sequence")

    # 1) Mod0: aligned to patch_size with neutral whitespace padding if odd
    mod0_pad = compute_ast_sync_pad(len(base), patch_size)
    mod0_toks = list(base) + ([" "] * mod0_pad)
    mod0 = SyncedSequence(
        condition="Mod0",
        raw_tokens=list(base),
        model_tokens=mod0_toks,
        structural_mask=[False] * len(mod0_toks),
        patch_size=patch_size,
        block_size=block_size,
        leading_shift=0,
        trailing_pad=mod0_pad,
        is_mod0_aligned=(len(mod0_toks) % patch_size == 0),
        source_ast_valid=ast_ok,
    )

    # 2) Mod1: simulate odd prompt boundary causing 1-token patch phase shift
    mod1_raw = ["<SHIFT>"] + list(base)
    mod1_tail = compute_ast_sync_pad(len(mod1_raw), patch_size)
    mod1_toks = mod1_raw + (["<PAD>"] * mod1_tail)
    mod1 = SyncedSequence(
        condition="Mod1",
        raw_tokens=list(base),
        model_tokens=mod1_toks,
        structural_mask=[False] * len(mod1_toks),
        patch_size=patch_size,
        block_size=block_size,
        leading_shift=1,
        trailing_pad=mod1_tail,
        is_mod0_aligned=False,
        source_ast_valid=ast_ok,
    )

    # 3) AST-Sync: align to patch_size with <AST_PAD> and protect AST skeleton tokens
    sync_pad = compute_ast_sync_pad(len(base), patch_size)
    sync_toks = list(base) + (["<AST_PAD>"] * sync_pad)
    sync_mask = [is_structural_ast_token(t) for t in base] + ([True] * sync_pad)
    ast_sync = SyncedSequence(
        condition="AST-Sync",
        raw_tokens=list(base),
        model_tokens=sync_toks,
        structural_mask=sync_mask,
        patch_size=patch_size,
        block_size=block_size,
        leading_shift=0,
        trailing_pad=sync_pad,
        is_mod0_aligned=(len(sync_toks) % patch_size == 0),
        source_ast_valid=ast_ok,
    )

    return [mod0, mod1, ast_sync]

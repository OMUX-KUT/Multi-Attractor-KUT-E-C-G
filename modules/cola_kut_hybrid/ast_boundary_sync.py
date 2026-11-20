"""Reversible Python lexer, AST verifier, 4-class token role classifier, and scope extractor."""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass
from typing import List, Literal, Set, Tuple

ConditionType = Literal["Mod0", "Mod1", "AST-Sync"]
RoleType = Literal["STRUCT", "IDENT_DECL", "TYPE_NAME", "EXPR_ATOM"]

_REVERSIBLE_TOKEN_RE = re.compile(
    r"\s+|[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|->|//|\S"
)

_NUMBER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")

_STRUCTURAL_SYMBOLS = {
    "(", ")", "[", "]", "{", "}", ":", ",", ".", "=", "+", "-", "*", "/", "//",
    "==", "!=", "<=", ">=", "->", "<", ">", "%",
}

BUILTIN_SAFE_CALLABLES: Set[str] = {"abs", "min", "max"}
VALID_TYPE_NAMES: Set[str] = {"int", "float", "bool", "str"}


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
    """Losslessly split Python source (including scientific notation like 1e-9)."""
    tokens = _REVERSIBLE_TOKEN_RE.findall(source)
    if "".join(tokens) != source:
        raise AssertionError("Reversible lexer invariant violated")
    return tokens


def is_structural_ast_token(tok: str) -> bool:
    if tok.isspace():
        return True
    if keyword.iskeyword(tok):
        return True
    if tok in _STRUCTURAL_SYMBOLS:
        return True
    return False


def is_valid_identifier(tok: str) -> bool:
    return tok.isidentifier() and (not keyword.iskeyword(tok))


def is_valid_number(tok: str) -> bool:
    return bool(_NUMBER_RE.fullmatch(tok))


def classify_token_roles(raw_tokens: List[str]) -> Tuple[List[RoleType], Set[str]]:
    """Assign each token to {STRUCT, IDENT_DECL, TYPE_NAME, EXPR_ATOM} and extract local params."""
    roles: List[RoleType] = []
    local_params: Set[str] = set()
    in_sig = False
    in_params = False
    prev_nonspace = ""

    for tok in raw_tokens:
        if tok == "def":
            in_sig = True
            in_params = False
            roles.append("STRUCT")
            prev_nonspace = tok
            continue

        if is_structural_ast_token(tok):
            roles.append("STRUCT")
            if in_sig:
                if tok == "(":
                    in_params = True
                elif tok == ")":
                    in_params = False
                elif tok == ":" and not in_params:
                    in_sig = False
            if not tok.isspace():
                prev_nonspace = tok
            continue

        # Non-structural token classification
        if in_sig:
            if prev_nonspace == "def":
                roles.append("IDENT_DECL")
            elif in_params and prev_nonspace in ("(", ","):
                roles.append("IDENT_DECL")
                local_params.add(tok)
            elif prev_nonspace in (":", "->"):
                roles.append("TYPE_NAME")
            else:
                roles.append("EXPR_ATOM")
        else:
            roles.append("EXPR_ATOM")

        prev_nonspace = tok

    return roles, local_params


def verify_ast_integrity(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def verify_scope_integrity(source: str, allowed_builtins: Set[str] = BUILTIN_SAFE_CALLABLES) -> bool:
    """Verify valid AST AND zero undefined identifier references (NameError prevention)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    defined_globals: Set[str] = set(allowed_builtins)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defined_globals.add(node.name)
            local_args = {arg.arg for arg in node.args.args}
            # Validate annotations use valid type names
            for arg in node.args.args:
                if isinstance(arg.annotation, ast.Name):
                    if arg.annotation.id not in VALID_TYPE_NAMES:
                        return False
            if isinstance(node.returns, ast.Name):
                if node.returns.id not in VALID_TYPE_NAMES:
                    return False
            # Validate all Name(Load) references inside function body are in scope
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id not in local_args and sub.id not in defined_globals and sub.id not in VALID_TYPE_NAMES:
                        return False
    return True


def compute_ast_sync_pad(length: int, patch_size: int) -> int:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    return (patch_size - (length % patch_size)) % patch_size


def build_three_conditions(
    source: str,
    patch_size: int = 2,
    block_size: int = 16,
) -> List[SyncedSequence]:
    ast_ok = verify_ast_integrity(source)
    base = reversible_tokenize(source)
    if not base:
        raise ValueError("Empty token sequence")

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

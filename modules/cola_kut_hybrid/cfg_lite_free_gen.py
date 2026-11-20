"""Phase 2B: Template-free open-ended Python generation with CFG-Lite incremental role estimator."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from modules.cola_kut_hybrid.ast_boundary_sync import (
    verify_ast_integrity,
    verify_scope_integrity,
)

FUNC_NAMES = ["f_ricci", "f_energy", "f_gate", "f_damp", "f_blend", "f_norm"]
PARAM_POOL = ["a", "b", "c", "x", "dt", "w", "lim", "rate"]
TYPE_POOL = ["int", "float", "bool"]
NUM_POOL = ["0.0", "1.0", "2.0", "1e-9", "500.0", "3840000"]
CALL_1 = ["abs"]
CALL_2 = ["min", "max"]
BIN_OPS = ["+", "-", "*", "/", "//", "%", "<=", ">=", "==", "and"]
SPECIAL_CTRL = ["def", "(", ")", ":", ",", "->", "return", "<EOS>"]

ALL_VOCAB: List[str] = (
    FUNC_NAMES + PARAM_POOL + TYPE_POOL + NUM_POOL + CALL_1 + CALL_2 + BIN_OPS + SPECIAL_CTRL
)
TOK2ID: Dict[str, int] = {t: i for i, t in enumerate(ALL_VOCAB)}
ID2TOK: Dict[int, str] = {i: t for i, t in enumerate(ALL_VOCAB)}


class CFGLiteStateMachine:
    """Zero-template incremental syntax & scope role estimator for open-ended function generation."""

    def __init__(self, max_params: int = 3, max_expr_tokens: int = 18, max_depth: int = 3) -> None:
        self.max_params = max_params
        self.max_expr_tokens = max_expr_tokens
        self.max_depth = max_depth
        self.state = "EXPECT_DEF"
        self.declared_params: List[str] = []
        self.pending_call_arity = 0
        self.stack: List[Tuple[str, int]] = []
        self.expr_steps_used = 0
        self.last_was_unary = False

    def _stack_close_cost(self) -> int:
        cost = 0
        for kind, rem in self.stack:
            cost += 1 if kind == "paren" else (2 * rem + 1)
        return cost

    def allowed_tokens(self) -> List[str]:
        s = self.state
        if s == "EXPECT_DEF":
            return ["def"]
        if s == "EXPECT_FUNC_NAME":
            return list(FUNC_NAMES)
        if s == "EXPECT_LPAREN":
            return ["("]
        if s == "EXPECT_PARAM_NAME":
            return [p for p in PARAM_POOL if p not in self.declared_params]
        if s == "EXPECT_PARAM_COLON":
            return [":"]
        if s == "EXPECT_PARAM_TYPE":
            return list(TYPE_POOL)
        if s == "EXPECT_COMMA_OR_RPAREN":
            if len(self.declared_params) < self.max_params and len(self.declared_params) < len(PARAM_POOL):
                return [",", ")"]
            return [")"]
        if s == "EXPECT_ARROW":
            return ["->"]
        if s == "EXPECT_RET_TYPE":
            return list(TYPE_POOL)
        if s == "EXPECT_DEF_COLON":
            return [":"]
        if s == "EXPECT_RETURN_KW":
            return ["return"]

        rem_steps = self.max_expr_tokens - self.expr_steps_used
        close_cost = self._stack_close_cost()

        if s == "EXPECT_CALL_LPAREN":
            return ["("]

        if s == "EXPECT_OPERAND":
            allowed: List[str] = list(self.declared_params) + list(NUM_POOL)
            if len(self.stack) < self.max_depth:
                if rem_steps - 1 >= close_cost + 2:
                    allowed.append("(")
                if rem_steps - 1 >= close_cost + 3:
                    allowed.extend(CALL_1)
                if rem_steps - 1 >= close_cost + 5:
                    allowed.extend(CALL_2)
            if (not self.last_was_unary) and (rem_steps - 1 >= close_cost + 1):
                allowed.append("-")
            return allowed

        if s == "EXPECT_AFTER_OPERAND":
            allowed = []
            if self.stack:
                kind, rem = self.stack[-1]
                allowed.append("," if (kind == "call" and rem > 0) else ")")
            else:
                allowed.append("<EOS>")
            if rem_steps - 1 >= close_cost + 1:
                allowed.extend(BIN_OPS)
            return allowed

        return ["<EOS>"]

    def step(self, tok: str) -> None:
        s = self.state
        if s == "EXPECT_DEF":
            self.state = "EXPECT_FUNC_NAME"
        elif s == "EXPECT_FUNC_NAME":
            self.state = "EXPECT_LPAREN"
        elif s == "EXPECT_LPAREN":
            self.state = "EXPECT_PARAM_NAME"
        elif s == "EXPECT_PARAM_NAME":
            self.declared_params.append(tok)
            self.state = "EXPECT_PARAM_COLON"
        elif s == "EXPECT_PARAM_COLON":
            self.state = "EXPECT_PARAM_TYPE"
        elif s == "EXPECT_PARAM_TYPE":
            self.state = "EXPECT_COMMA_OR_RPAREN"
        elif s == "EXPECT_COMMA_OR_RPAREN":
            self.state = "EXPECT_PARAM_NAME" if tok == "," else "EXPECT_ARROW"
        elif s == "EXPECT_ARROW":
            self.state = "EXPECT_RET_TYPE"
        elif s == "EXPECT_RET_TYPE":
            self.state = "EXPECT_DEF_COLON"
        elif s == "EXPECT_DEF_COLON":
            self.state = "EXPECT_RETURN_KW"
        elif s == "EXPECT_RETURN_KW":
            self.state = "EXPECT_OPERAND"
        elif s == "EXPECT_CALL_LPAREN":
            self.expr_steps_used += 1
            self.stack.append(("call", self.pending_call_arity - 1))
            self.last_was_unary = False
            self.state = "EXPECT_OPERAND"
        elif s == "EXPECT_OPERAND":
            self.expr_steps_used += 1
            if tok == "(":
                self.stack.append(("paren", 0))
                self.last_was_unary = False
                self.state = "EXPECT_OPERAND"
            elif tok == "-":
                self.last_was_unary = True
                self.state = "EXPECT_OPERAND"
            elif tok in CALL_1:
                self.pending_call_arity = 1
                self.last_was_unary = False
                self.state = "EXPECT_CALL_LPAREN"
            elif tok in CALL_2:
                self.pending_call_arity = 2
                self.last_was_unary = False
                self.state = "EXPECT_CALL_LPAREN"
            else:
                self.last_was_unary = False
                self.state = "EXPECT_AFTER_OPERAND"
        elif s == "EXPECT_AFTER_OPERAND":
            if tok == "<EOS>":
                self.state = "DONE"
                return
            self.expr_steps_used += 1
            if tok == ",":
                kind, rem = self.stack.pop()
                self.stack.append((kind, rem - 1))
                self.last_was_unary = False
                self.state = "EXPECT_OPERAND"
            elif tok == ")":
                self.stack.pop()
                self.last_was_unary = False
                self.state = "EXPECT_AFTER_OPERAND"
            else:
                self.last_was_unary = False
                self.state = "EXPECT_OPERAND"


def render_generated_tokens(tokens: List[str]) -> str:
    if "return" not in tokens:
        return " ".join(t for t in tokens if t != "<EOS>")
    ret_idx = tokens.index("return")
    header = tokens[:ret_idx]
    body = [t for t in tokens[ret_idx + 1 :] if t != "<EOS>"]

    out_h: List[str] = []
    for i, t in enumerate(header):
        if i > 0 and t not in ("(", ")", ",", ":") and header[i - 1] not in ("(",):
            out_h.append(" ")
        out_h.append(t)
    return "".join(out_h) + "\n    return " + " ".join(body) + "\n"


@dataclass(frozen=True)
class FreeGenRow:
    noise_std: float
    mode: str
    generations: int
    avg_tokens: float
    ast_corruption_rate: float
    scope_error_rate: float
    measured_tps: float
    sample_preview: str


def run_free_gen_benchmark(
    noise_grid: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    n_gens_per_noise: int = 50,
    seed: int = 2026,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    proj = torch.randn(110, len(ALL_VOCAB)) * 0.25
    rows: List[FreeGenRow] = []

    for noise_std in noise_grid:
        for mode in ("Unconstrained-FreeGen", "CFG-Lite-FreeGen"):
            ast_err = 0
            scope_err = 0
            total_toks = 0
            preview = ""
            t0 = time.perf_counter()

            for g in range(n_gens_per_noise):
                torch.manual_seed(seed + int(noise_std * 100) * 1000 + g)
                sm = CFGLiteStateMachine(max_params=3, max_expr_tokens=18, max_depth=3)
                latents = torch.randn(40, 110) * (1.0 + noise_std)
                step_logits = latents @ proj

                gen_tokens: List[str] = []
                for step_idx in range(40):
                    logits = step_logits[step_idx]
                    if mode == "CFG-Lite-FreeGen":
                        allowed = sm.allowed_tokens()
                        mask = torch.zeros(len(ALL_VOCAB), dtype=torch.bool)
                        for tok in allowed:
                            mask[TOK2ID[tok]] = True
                        chosen_id = int(torch.argmax(logits.masked_fill(~mask, -1e9)).item())
                        tok = ID2TOK[chosen_id]
                        sm.step(tok)
                        if tok == "<EOS>" or sm.state == "DONE":
                            break
                        gen_tokens.append(tok)
                    else:
                        if sm.state not in ("EXPECT_OPERAND", "EXPECT_AFTER_OPERAND", "EXPECT_CALL_LPAREN", "DONE"):
                            allowed = sm.allowed_tokens()
                            mask = torch.zeros(len(ALL_VOCAB), dtype=torch.bool)
                            for tok in allowed:
                                mask[TOK2ID[tok]] = True
                            chosen_id = int(torch.argmax(logits.masked_fill(~mask, -1e9)).item())
                            tok = ID2TOK[chosen_id]
                            sm.step(tok)
                            gen_tokens.append(tok)
                        else:
                            chosen_id = int(torch.argmax(logits).item())
                            tok = ID2TOK[chosen_id]
                            if tok == "<EOS>":
                                break
                            gen_tokens.append(tok)

                total_toks += len(gen_tokens)
                src = render_generated_tokens(gen_tokens)
                if not verify_ast_integrity(src):
                    ast_err += 1
                if not verify_scope_integrity(src):
                    scope_err += 1
                if not preview:
                    preview = src.replace("\n", "\\n")[:130]

            elapsed = max(1e-9, time.perf_counter() - t0)
            rows.append(
                FreeGenRow(
                    noise_std=noise_std,
                    mode=mode,
                    generations=n_gens_per_noise,
                    avg_tokens=round(total_toks / n_gens_per_noise, 2),
                    ast_corruption_rate=round(ast_err / n_gens_per_noise, 6),
                    scope_error_rate=round(scope_err / n_gens_per_noise, 6),
                    measured_tps=round(total_toks / elapsed, 2),
                    sample_preview=preview,
                )
            )

    cfg_rows = [r for r in rows if r.mode == "CFG-Lite-FreeGen"]
    phase2b_zero_ast = all(r.ast_corruption_rate == 0.0 for r in cfg_rows)
    phase2b_zero_scope = all(r.scope_error_rate == 0.0 for r in cfg_rows)
    phase2b_tps_ok = all(r.measured_tps >= 500.0 for r in cfg_rows)
    phase2b_pass = phase2b_zero_ast and phase2b_zero_scope and phase2b_tps_ok

    report = {
        "total_cfg_lite_generations": len(cfg_rows) * n_gens_per_noise,
        "phase2b_zero_ast_err": phase2b_zero_ast,
        "phase2b_zero_scope_err": phase2b_zero_scope,
        "phase2b_tps_ge_500": phase2b_tps_ok,
        "phase2b_pass": phase2b_pass,
        "verdict": "PASS_PHASE2B_CFG_LITE_FREEGEN" if phase2b_pass else "FALSIFIED_SWITCH_FULL_PDA_OR_CRAM",
        "rows": [asdict(r) for r in rows],
    }
    out_path = Path("logs/phase2b_cfg_lite_freegen.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run_free_gen_benchmark()
    print(f"=== PHASE 2B CFG-LITE FREE-GEN AUDIT | VERDICT={rep['verdict']} ===")
    for r in rep["rows"]:
        print(
            f"[noise={r['noise_std']:3.1f}][{r['mode']:21s}] "
            f"avg_tok={r['avg_tokens']:5.1f} | AST_err={r['ast_corruption_rate']:6.1%} | "
            f"Scope_err={r['scope_error_rate']:6.1%} | TPS={r['measured_tps']:9.1f}"
        )
        if r["mode"] == "CFG-Lite-FreeGen" and r["noise_std"] == 1.0:
            print(f"   sample: {r['sample_preview']}")

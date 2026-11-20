"""Phase 3 Option A: Typed-CFG-Lite with strict static typechecker & sandbox runtime verification."""

from __future__ import annotations

import ast
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from modules.cola_kut_hybrid.ast_boundary_sync import (
    verify_ast_integrity,
    verify_scope_integrity,
)
from modules.cola_kut_hybrid.cfg_lite_free_gen import (
    ALL_VOCAB as P2_VOCAB,
    CFGLiteStateMachine,
    ID2TOK as P2_ID2TOK,
    TOK2ID as P2_TOK2ID,
    render_generated_tokens as render_p2_tokens,
)

FUNC_NAMES = ["f_ricci", "f_energy", "f_gate", "f_damp", "f_blend", "f_norm"]
PARAM_POOL = ["a", "b", "c", "x", "dt", "w", "lim", "rate"]
TYPES = ["int", "float", "bool"]
INT_LITS = ["0", "1", "2", "16", "500"]
FLOAT_LITS = ["0.0", "1.0", "2.0", "1e-9", "500.0"]
BOOL_LITS = ["True", "False"]

INT_RULES = ["atom", "abs", "neg", "add", "sub", "mul", "safe_floordiv", "safe_mod", "min", "max"]
FLOAT_RULES = ["atom", "abs", "neg", "add", "sub", "mul", "safe_div", "min", "max"]
BOOL_RULES = ["atom", "not", "and", "or", "cmp_int_le", "cmp_int_eq", "cmp_float_le", "cmp_float_ge"]

DECISION_VOCAB: List[str] = (
    FUNC_NAMES + PARAM_POOL + TYPES + INT_LITS + FLOAT_LITS + BOOL_LITS
    + ["more_param", "end_params"]
    + [f"int:{r}" for r in INT_RULES]
    + [f"float:{r}" for r in FLOAT_RULES]
    + [f"bool:{r}" for r in BOOL_RULES]
)
D2ID: Dict[str, int] = {k: i for i, k in enumerate(DECISION_VOCAB)}


def _is_guaranteed_nonzero_ast(node: ast.AST, expected_ty: str) -> bool:
    """Verify denominator AST structurally guarantees non-zero value (e.g., abs(x) + c with c > 0)."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value) != 0.0
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # abs(...) + positive_const OR positive_const + abs(...)
        for a, b in ((node.left, node.right), (node.right, node.left)):
            is_abs = (
                isinstance(a, ast.Call)
                and isinstance(a.func, ast.Name)
                and a.func.id == "abs"
                and len(a.args) == 1
            )
            is_pos_const = (
                isinstance(b, ast.Constant)
                and (type(b.value) is int if expected_ty == "int" else type(b.value) is float)
                and float(b.value) > 0.0
            )
            if is_abs and is_pos_const:
                return True
    return False


def strict_static_typecheck(source: str) -> Tuple[bool, str]:
    """Independent bottom-up strict type checker treating bool != int and requiring safe denominators."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "Expected single FunctionDef"

    fn = tree.body[0]
    env: Dict[str, str] = {}
    for arg in fn.args.args:
        if not isinstance(arg.annotation, ast.Name) or arg.annotation.id not in ("int", "float", "bool"):
            return False, f"Invalid param annotation for {arg.arg}"
        env[arg.arg] = arg.annotation.id

    if not isinstance(fn.returns, ast.Name) or fn.returns.id not in ("int", "float", "bool"):
        return False, "Invalid return annotation"
    ret_ty = fn.returns.id

    def infer(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant):
            if type(node.value) is bool:
                return "bool"
            if type(node.value) is int:
                return "int"
            if type(node.value) is float:
                return "float"
            return None
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.UnaryOp):
            t = infer(node.operand)
            if isinstance(node.op, ast.Not):
                return "bool" if t == "bool" else None
            if isinstance(node.op, ast.USub):
                return t if t in ("int", "float") else None
            return None
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                return None
            for v in node.values:
                if infer(v) != "bool":
                    return None
            return "bool"
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                return None
            lt = infer(node.left)
            rt = infer(node.comparators[0])
            if lt in ("int", "float") and lt == rt:
                return "bool"
            return None
        if isinstance(node, ast.BinOp):
            lt = infer(node.left)
            rt = infer(node.right)
            if lt != rt:
                return None
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                return lt if lt in ("int", "float") else None
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                if lt == "int" and _is_guaranteed_nonzero_ast(node.right, "int"):
                    return "int"
                return None
            if isinstance(node.op, ast.Div):
                if lt == "float" and _is_guaranteed_nonzero_ast(node.right, "float"):
                    return "float"
                return None
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return None
            fname = node.func.id
            if fname == "abs" and len(node.args) == 1:
                t = infer(node.args[0])
                return t if t in ("int", "float") else None
            if fname in ("min", "max") and len(node.args) == 2:
                t1, t2 = infer(node.args[0]), infer(node.args[1])
                return t1 if (t1 in ("int", "float") and t1 == t2) else None
            return None
        return None

    if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return) or fn.body[0].value is None:
        return False, "Expected single Return statement"

    inferred = infer(fn.body[0].value)
    if inferred != ret_ty:
        return False, f"Return type mismatch: annotated={ret_ty}, inferred={inferred}"
    return True, "ok"


def sandbox_runtime_test(source: str) -> Tuple[bool, str]:
    """Execute generated function in restricted namespace over boundary inputs (0, -1, 0.0, False, True)."""
    try:
        tree = ast.parse(source)
        fn_def = tree.body[0]
        assert isinstance(fn_def, ast.FunctionDef)
        param_types = [arg.annotation.id for arg in fn_def.args.args]  # type: ignore[union-attr]
        ret_type_name = fn_def.returns.id  # type: ignore[union-attr]
    except Exception as exc:
        return False, f"Parse/Meta error: {exc}"

    safe_globals: Dict[str, Any] = {
        "__builtins__": {},
        "abs": abs,
        "min": min,
        "max": max,
    }
    local_ns: Dict[str, Any] = {}
    try:
        exec(compile(tree, "<typed_cfg_sandbox>", "exec"), safe_globals, local_ns)
        fn = local_ns[fn_def.name]
    except Exception as exc:
        return False, f"Compile/Exec error: {exc}"

    domain: Dict[str, List[Any]] = {
        "int": [0, -1, 2],
        "float": [0.0, -1.0, 1.5],
        "bool": [False, True],
    }
    expected_py_type = {"int": int, "float": float, "bool": bool}[ret_type_name]
    arg_lists = [domain[t] for t in param_types]

    for args in itertools.product(*arg_lists):
        try:
            out = fn(*args)
        except Exception as exc:
            return False, f"Runtime exception {type(exc).__name__} on args={args}: {exc}"
        if type(out) is not expected_py_type:
            return False, f"Runtime type mismatch: got {type(out).__name__}, expected {ret_type_name}"
        if expected_py_type is float and (not math.isfinite(out)):
            return False, f"Non-finite float result {out} on args={args}"

    return True, "ok"


class TypedCFGLiteGenerator:
    """Type-directed incremental state-machine decoder with guaranteed non-zero denominators."""

    def __init__(self, step_logits: torch.Tensor, max_params: int = 3, max_depth: int = 3) -> None:
        self.step_logits = step_logits
        self.cursor = 0
        self.max_params = max_params
        self.max_depth = max_depth
        self.token_count = 0

    def _choose(self, candidates: List[str]) -> str:
        row = self.step_logits[self.cursor % self.step_logits.shape[0]]
        self.cursor += 1
        best_c = candidates[0]
        best_score = -1e18
        for c in candidates:
            sc = float(row[D2ID[c]].item())
            if sc > best_score:
                best_score = sc
                best_c = c
        return best_c

    def _gen_expr(self, target_ty: str, depth: int, env_by_ty: Dict[str, List[str]]) -> List[str]:
        if target_ty == "int":
            atoms = env_by_ty["int"] + INT_LITS
            rule_pool = ["int:atom"] if depth <= 0 else [f"int:{r}" for r in INT_RULES]
            rule = self._choose(rule_pool).split(":", 1)[1]
            if rule == "atom":
                return [self._choose(atoms)]
            if rule == "abs":
                return ["abs", "("] + self._gen_expr("int", depth - 1, env_by_ty) + [")"]
            if rule == "neg":
                return ["(", "-"] + self._gen_expr("int", depth - 1, env_by_ty) + [")"]
            if rule in ("add", "sub", "mul"):
                op = {"add": "+", "sub": "-", "mul": "*"}[rule]
                return ["("] + self._gen_expr("int", depth - 1, env_by_ty) + [op] + self._gen_expr("int", depth - 1, env_by_ty) + [")"]
            if rule in ("safe_floordiv", "safe_mod"):
                op = "//" if rule == "safe_floordiv" else "%"
                lhs = self._gen_expr("int", depth - 1, env_by_ty)
                den = self._gen_expr("int", depth - 1, env_by_ty)
                # Guaranteed non-zero denominator: ( abs ( den ) + 1 ) >= 1
                return ["("] + lhs + [op, "(", "abs", "("] + den + [")", "+", "1", ")", ")"]
            fn_name = rule  # min / max
            return [fn_name, "("] + self._gen_expr("int", depth - 1, env_by_ty) + [","] + self._gen_expr("int", depth - 1, env_by_ty) + [")"]

        if target_ty == "float":
            atoms = env_by_ty["float"] + FLOAT_LITS
            rule_pool = ["float:atom"] if depth <= 0 else [f"float:{r}" for r in FLOAT_RULES]
            rule = self._choose(rule_pool).split(":", 1)[1]
            if rule == "atom":
                return [self._choose(atoms)]
            if rule == "abs":
                return ["abs", "("] + self._gen_expr("float", depth - 1, env_by_ty) + [")"]
            if rule == "neg":
                return ["(", "-"] + self._gen_expr("float", depth - 1, env_by_ty) + [")"]
            if rule in ("add", "sub", "mul"):
                op = {"add": "+", "sub": "-", "mul": "*"}[rule]
                return ["("] + self._gen_expr("float", depth - 1, env_by_ty) + [op] + self._gen_expr("float", depth - 1, env_by_ty) + [")"]
            if rule == "safe_div":
                lhs = self._gen_expr("float", depth - 1, env_by_ty)
                den = self._gen_expr("float", depth - 1, env_by_ty)
                # Guaranteed non-zero denominator: ( abs ( den ) + 1e-9 ) >= 1e-9
                return ["("] + lhs + ["/", "(", "abs", "("] + den + [")", "+", "1e-9", ")", ")"]
            fn_name = rule  # min / max
            return [fn_name, "("] + self._gen_expr("float", depth - 1, env_by_ty) + [","] + self._gen_expr("float", depth - 1, env_by_ty) + [")"]

        # target_ty == "bool"
        atoms = env_by_ty["bool"] + BOOL_LITS
        rule_pool = ["bool:atom"] if depth <= 0 else [f"bool:{r}" for r in BOOL_RULES]
        rule = self._choose(rule_pool).split(":", 1)[1]
        if rule == "atom":
            return [self._choose(atoms)]
        if rule == "not":
            return ["(", "not"] + self._gen_expr("bool", depth - 1, env_by_ty) + [")"]
        if rule in ("and", "or"):
            return ["("] + self._gen_expr("bool", depth - 1, env_by_ty) + [rule] + self._gen_expr("bool", depth - 1, env_by_ty) + [")"]
        if rule in ("cmp_int_le", "cmp_int_eq"):
            op = "<=" if rule == "cmp_int_le" else "=="
            return ["("] + self._gen_expr("int", depth - 1, env_by_ty) + [op] + self._gen_expr("int", depth - 1, env_by_ty) + [")"]
        op = "<=" if rule == "cmp_float_le" else ">="
        return ["("] + self._gen_expr("float", depth - 1, env_by_ty) + [op] + self._gen_expr("float", depth - 1, env_by_ty) + [")"]

    def generate(self) -> Tuple[str, int]:
        fn_name = self._choose(FUNC_NAMES)
        used_params: List[str] = []
        param_decls: List[Tuple[str, str]] = []
        env_by_ty: Dict[str, List[str]] = {"int": [], "float": [], "bool": []}

        for idx in range(self.max_params):
            avail = [p for p in PARAM_POOL if p not in used_params]
            pname = self._choose(avail)
            pty = self._choose(TYPES)
            used_params.append(pname)
            param_decls.append((pname, pty))
            env_by_ty[pty].append(pname)
            if idx + 1 < self.max_params:
                if self._choose(["more_param", "end_params"]) == "end_params":
                    break

        ret_ty = self._choose(TYPES)
        body_tokens = self._gen_expr(ret_ty, self.max_depth, env_by_ty)

        sig_str = ", ".join(f"{n}: {t}" for n, t in param_decls)
        src = f"def {fn_name}({sig_str}) -> {ret_ty}:\n    return {' '.join(body_tokens)}\n"
        # Count emitted tokens (header + return body)
        total_tokens = 6 + len(param_decls) * 4 + len(body_tokens)
        return src, total_tokens


@dataclass(frozen=True)
class Phase3Row:
    noise_std: float
    mode: str
    generations: int
    avg_tokens: float
    ast_err_rate: float
    scope_err_rate: float
    type_err_rate: float
    runtime_err_rate: float
    measured_tps: float
    sample_preview: str


def run_phase3_benchmark(
    noise_grid: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    n_gens_per_noise: int = 50,
    seed: int = 2026,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    p2_proj = torch.randn(110, len(P2_VOCAB)) * 0.25
    p3_proj = torch.randn(110, len(DECISION_VOCAB)) * 0.25

    rows: List[Phase3Row] = []

    for noise_std in noise_grid:
        for mode in ("Phase2B-CFG-Lite", "Phase3-Typed-CFG-Lite"):
            ast_err = scope_err = type_err = rt_err = 0
            total_toks = 0
            preview = ""
            t0 = time.perf_counter()

            for g in range(n_gens_per_noise):
                torch.manual_seed(seed + int(noise_std * 100) * 1000 + g)
                latents = torch.randn(64, 110) * (1.0 + noise_std)

                if mode == "Phase2B-CFG-Lite":
                    step_logits = latents @ p2_proj
                    sm = CFGLiteStateMachine(max_params=3, max_expr_tokens=18, max_depth=3)
                    gen_tokens: List[str] = []
                    for step_idx in range(40):
                        allowed = sm.allowed_tokens()
                        mask = torch.zeros(len(P2_VOCAB), dtype=torch.bool)
                        for tok in allowed:
                            mask[P2_TOK2ID[tok]] = True
                        cid = int(torch.argmax(step_logits[step_idx].masked_fill(~mask, -1e9)).item())
                        tok = P2_ID2TOK[cid]
                        sm.step(tok)
                        if tok == "<EOS>" or sm.state == "DONE":
                            break
                        gen_tokens.append(tok)
                    src = render_p2_tokens(gen_tokens)
                    ntok = len(gen_tokens)
                else:
                    step_logits = latents @ p3_proj
                    gen = TypedCFGLiteGenerator(step_logits, max_params=3, max_depth=3)
                    src, ntok = gen.generate()

                total_toks += ntok
                if not verify_ast_integrity(src):
                    ast_err += 1
                if not verify_scope_integrity(src):
                    scope_err += 1
                ok_ty, _ = strict_static_typecheck(src)
                if not ok_ty:
                    type_err += 1
                ok_rt, _ = sandbox_runtime_test(src)
                if not ok_rt:
                    rt_err += 1
                if not preview:
                    preview = src.replace("\n", "\\n")[:150]

            elapsed = max(1e-9, time.perf_counter() - t0)
            rows.append(
                Phase3Row(
                    noise_std=noise_std,
                    mode=mode,
                    generations=n_gens_per_noise,
                    avg_tokens=round(total_toks / n_gens_per_noise, 2),
                    ast_err_rate=round(ast_err / n_gens_per_noise, 6),
                    scope_err_rate=round(scope_err / n_gens_per_noise, 6),
                    type_err_rate=round(type_err / n_gens_per_noise, 6),
                    runtime_err_rate=round(rt_err / n_gens_per_noise, 6),
                    measured_tps=round(total_toks / elapsed, 2),
                    sample_preview=preview,
                )
            )

    p3_rows = [r for r in rows if r.mode == "Phase3-Typed-CFG-Lite"]
    zero_ast = all(r.ast_err_rate == 0.0 for r in p3_rows)
    zero_scope = all(r.scope_err_rate == 0.0 for r in p3_rows)
    zero_type = all(r.type_err_rate == 0.0 for r in p3_rows)
    zero_rt = all(r.runtime_err_rate == 0.0 for r in p3_rows)
    tps_ok = all(r.measured_tps >= 500.0 for r in p3_rows)
    p3_pass = zero_ast and zero_scope and zero_type and zero_rt and tps_ok

    report = {
        "total_typed_generations": len(p3_rows) * n_gens_per_noise,
        "phase3_zero_ast_err": zero_ast,
        "phase3_zero_scope_err": zero_scope,
        "phase3_zero_type_err": zero_type,
        "phase3_zero_runtime_err": zero_rt,
        "phase3_tps_ge_500": tps_ok,
        "phase3_pass": p3_pass,
        "migrate_to_bidirectional_typechecker": not p3_pass,
        "verdict": "PASS_PHASE3_TYPED_CFG_LITE_4FOLD_SAFE" if p3_pass else "FALSIFIED_MIGRATE_BIDIRECTIONAL_TYPECHECKER",
        "rows": [asdict(r) for r in rows],
    }
    Path("logs/phase3_typed_cfg_lite.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    rep = run_phase3_benchmark()
    print(f"=== PHASE 3 TYPED-CFG-LITE AUDIT | VERDICT={rep['verdict']} ===")
    for r in rep["rows"]:
        print(
            f"[noise={r['noise_std']:3.1f}][{r['mode']:21s}] "
            f"AST={r['ast_err_rate']:5.1%} | Scope={r['scope_err_rate']:5.1%} | "
            f"Type={r['type_err_rate']:5.1%} | Runtime={r['runtime_err_rate']:5.1%} | TPS={r['measured_tps']:8.1f}"
        )
        if r["mode"] == "Phase3-Typed-CFG-Lite" and r["noise_std"] == 1.0:
            print(f"   sample: {r['sample_preview']}")

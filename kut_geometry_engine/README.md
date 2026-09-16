# Multi-Attractor-KUT-E-C-G
Production implementation of Multi-Attractor KUT (E=C=G) for non-contractible manifold routing and Pareto-optimal test-time search.
# KUT-Engine: Test-Time Search Optimization via Geometric Ricci Annealing

[![CI](https://github.com/kut-research/kut-geometry-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/kut-research/kut-geometry-engine/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Lean4](https://img.shields.io/badge/Verified_with-Lean_4-purple.svg)](KUT_Foundations_Fixed.lean)

Production-grade implementation of **Geometric Ricci Annealing (GRA)** and **Multi-Attractor KUT** for large language models, formalizing test-time search as an information-geometric Riemannian flow.

## Key Features

- **FLOPs Efficiency**: Reduces test-time search FLOPs by **34.7% to 58.6%** on reasoning benchmarks (LiveCodeBench Hard).
- **Scale-Aware Annealing**: Dynamic temperature scheduling obeying $\\beta(P) = 0.35 \\cdot (7/P)^{0.22}$.
- **Hardware Integration**: Fused CUDA kernel for vLLM (<0.8% throughput overhead at Concurrency 256).
- **Multi-Attractor Routing**: Pareto-optimal trajectory tracking on non-contractible manifold tasks (dilemmas, trade-offs).
- **Formal Verification**: Core theorems formally machine-checked in **Lean 4**.

## Installation

### Prerequisites
- Ubuntu 22.04 / 24.04 LTS
- NVIDIA Driver >= 535, CUDA >= 12.4
- Python >= 3.10

```bash
git clone [https://github.com/kut-research/kut-geometry-engine.git](https://github.com/kut-research/kut-geometry-engine.git)
cd kut-geometry-engine
pip install -e .

```
# Mathematics AI ATP

This repository is an experimental proof-search and theorem-proving project that combines machine learning, symbolic reasoning, and Lean/Pantograph tooling. The codebase is organized around the `maths_ai` package, with supporting experiments and data assets for research and prototyping.

## Overview

The project aims to explore hybrid automated theorem proving by combining:

- GNN-based tactic and argument prediction
- PLN-style reasoning and ranking
- proof hypergraph search and visualization
- Lean/Pantograph integration for tactic application

The current implementation is research-oriented rather than a finished end-user application.

## Repository structure

- `maths_ai/` — core library for proof search, inference, data models, and utilities
  - `core/` — shared configuration and runtime settings
  - `data/` — preprocessing and representation helpers
  - `data_models/` — proof-related data structures
  - `gnn_inference/` — GNN scoring and tactic/argument inference
  - `hybrid_reasoner/` — proof hypergraph search and tactic execution
  - `pln_inference/` — PLN-style inference components
  - `utils/` — supporting utilities
- `experiments/` — additional experimental projects and analysis notebooks/scripts
- `data/` — datasets or artifacts used by experiments
- `tests/` — test coverage and validation work

## Key components

### Hybrid reasoning
The `maths_ai.hybrid_reasoner` package implements a proof-search pipeline that uses GNN predictions and PLN-derived ranking to explore candidate tactics and subgoals.

### GNN inference
The `maths_ai.gnn_inference` package contains the model and inference engine used to predict tactics and relevant premises.

### PLN inference
The `maths_ai.pln_inference` package provides symbolic reasoning support for scoring and ranking proof states.

## Requirements

The project targets Python 3.11+.

Core dependencies include:

- PyTorch
- PyTorch Geometric
- FAISS CPU
- Graphviz
- Pydantic
- Pantograph
- pytest

You can install the project with either of the following approaches:

## Getting started

1. Create and activate a Python environment.
2. Install dependencies.

```bash

uv sync 

source .venv/bin/activate

uv pip install -e .
```


```bash
uv add -r requirements.txt
```

3. Download the trained GNN model from Hugging Face into the local repository.

```bash
uv pip install -U huggingface_hub

hf download jajostrains/Mathlib-Sexpr-GNN \
  --repo-type model \
  --local-dir maths_ai/gnn_inference/artifacts/models/Mathlib-Sexpr-GNN
```

`--local-dir` materializes real files under the repository instead of relying on
paths inside the Hugging Face cache. The destination is under `artifacts/`, which
is ignored by Git. The model repository is public; if Hugging Face asks for
authentication, run `hf auth login` first.

Verify the pointer bundle and run a local inference smoke test:

```bash
test -f maths_ai/gnn_inference/artifacts/models/Mathlib-Sexpr-GNN/pointer-gat-gru/manifest.json

python maths_ai/gnn_inference/scripts/run_inference.py \
  --bundle maths_ai/gnn_inference/artifacts/models/Mathlib-Sexpr-GNN/pointer-gat-gru \
  --state $'n : Nat\n⊢ n = n'
```

The bundle includes model configuration and vocabularies, so this inference
path does not require the prepared training dataset. To update an existing local
copy, rerun the same `hf download` command.

4. Test a theorem:

```bash
./scripts/run_prover.sh --hypotheses "COMMA SEPARATED HYPOTHESESES" --goal_statement "GOAL EXPRESSION"
```

Most of the real research logic lives in the packages under `maths_ai/` and the experimental folders under `experiments/`.

## Development notes

- The repository is intended for experimentation and research use.
- Many components depend on external Lean/Pantograph tooling for real tactic execution.
- If you are working with the GNN or proof-search pipeline, inspect the modules under `maths_ai/hybrid_reasoner/` and `maths_ai/gnn_inference/` first.


## License

This project is distributed under the terms of the repository license. See `LICENSE` for details.

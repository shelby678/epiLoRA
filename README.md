# autoprot

Autonomous protein language model research, inspired by
[autoresearch](https://github.com/karpathy/autoresearch).

An AI coding agent (Claude Code, Codex, etc.) iteratively modifies `train.py`
to minimize validation loss on a protein masked language model, running
experiments in a tight loop with git-based version control.

## How it works

1. Point an autonomous coding agent at `program.md`
2. The agent modifies `train.py`, commits, runs the experiment, and evaluates
3. Improvements are kept; failures are reverted via `git reset`
4. Results are logged to `results.tsv`
5. The loop runs until you stop it

## Files

| File | Purpose |
|------|---------|
| `prepare.py` | **IMMUTABLE** — tokenizer, data loading, masking, evaluation |
| `train.py` | **MUTABLE** — model architecture, optimizer, training loop |
| `program.md` | Agent instructions and research objectives |

## Quick start

```bash
# Put FASTA files in data/train/ and data/val/
cp /path/to/train_sequences.fasta data/train/
cp /path/to/val_sequences.fasta data/val/

# Supports .fasta, .fa, .fasta.gz, and .fa.gz files

# Start the agent with program.md as context
# (e.g., in Claude Code, just point it to program.md)
```

## Manual run

```bash
uv run train.py
```

## B-cell epitope prediction (ESM3 + LoRA + RYS)

The main line of work is per-residue B-cell epitope prediction on antigen
sequences: **ESM3-small-open** (frozen) + **LoRA** + optional **RYS** block
replay + a linear head (`train_struct.py`). Experiments run as 3-fold CV on
`data/BEPIPRED.fasta` and log to `results.tsv`.

```bash
uv run python run_bepipred.py --list        # list experiment sets
uv run python run_bepipred.py baseline       # run one set
uv run python run_bepipred.py hiddenkey      # DropKey / HiddenCut / KL sweep
```

Other runners: `run_ensemble.py {esm3,esm2}` and `run_ensemble_esmif1.py`
(ensemble members), `run_if1_5fold.py {lora,xgb}` (5-fold IF1 comparison),
`run_bepipred_discotope*.py` (XGBoost-on-embeddings), `run_bepipred_esmc.py`
(ESMC backbone). Data-processing scripts live in `data/`. See `CLAUDE.md` for
the full layout.

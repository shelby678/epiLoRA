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

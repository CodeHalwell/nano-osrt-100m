---
name: hf-cli
description: >
  Use this skill when the user asks to interact with the Hugging Face Hub via
  the `hf` CLI. Triggers on: any mention of `hf` command, "huggingface CLI",
  "hf cli", "huggingface-cli" (deprecated alias), uploading or downloading
  models/datasets, listing/searching models or datasets on the Hub, managing
  HF Spaces, running HF Jobs (compute), managing HF tokens / auth, pushing a
  checkpoint to HF, pulling a dataset from HF, creating a new repo on the Hub,
  setting up HF cache, or any task involving HF repos, branches, or LFS.
  Also use when the user wants to install or update the agent skill for the
  `hf` CLI itself (`hf skills add`). Prefer this skill over guessing
  command syntax — the official CLI has the canonical reference.
---

# Hugging Face CLI

Connect agents to the Hugging Face Hub: search models, manage datasets and
buckets, launch Spaces, run jobs.

## Quick install check

Before invoking any `hf` command, confirm the CLI is installed and current:

```bash
hf --version
```

If missing or `command not found`, install via pip:

```bash
pip install -U "huggingface_hub[cli]"
```

(The user may also have it via Homebrew, uv, or conda — the
[official install guide](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started)
covers every path.)

## Authentication

Most write operations and many read operations on private content require
auth. Two ways:

```bash
# interactive (opens browser-ish prompt for token paste)
hf auth login

# non-interactive — reads from env (preferred for scripts/agents)
export HF_TOKEN="hf_..."          # get from https://huggingface.co/settings/tokens
hf whoami                          # confirm
```

If the user has a `HF_TOKEN` in their `.env`, source it before calling `hf`.
If they have multiple tokens, use `hf auth switch` to pick one.

## Most-used commands (cheat sheet)

| task | command |
|---|---|
| list logged-in user | `hf whoami` |
| search models | `hf models list --filter <tag> --limit 20` |
| list files in a repo | `hf repo files <user>/<repo>` |
| download a model | `hf download <user>/<repo>` or `--include "*.safetensors"` for partial |
| download a dataset | `hf download --repo-type dataset <user>/<repo>` |
| upload a file | `hf upload <user>/<repo> ./local/path [./remote/path]` |
| upload a folder | `hf upload <user>/<repo> ./local_dir --include="*.pt"` |
| create a repo | `hf repo create <user>/<repo> [--type model\|dataset\|space]` |
| set repo visibility | `hf repo update <user>/<repo> --private` |
| delete a file | `hf repo file delete <user>/<repo> <path>` |
| list cache | `hf cache scan` |
| clean cache | `hf cache delete --pattern "*"` |
| run a job | `hf jobs run <docker_image> --flavor a10g-large --command "..."` |
| list jobs | `hf jobs list` |
| view job logs | `hf jobs logs <job_id>` |
| install the agent skill | `hf skills add --claude --global` |

## Decision rules

1. **Never guess flag names** — `hf <command> --help` is fast and authoritative.
   Run it instead of inventing syntax.
2. **Repo type matters**. `hf download <id>` defaults to `--repo-type model`.
   For datasets or spaces, pass `--repo-type dataset` / `--repo-type space`.
3. **Use `--include` / `--exclude` patterns** for partial downloads. Pulling a
   full LLM repo when the user only needs the tokenizer wastes bandwidth.
4. **For large uploads** (>5 GB single file), check if LFS is set up on the
   repo. `hf upload` handles this automatically but verify success by
   re-running `hf repo files <user>/<repo>` after.
5. **Token scope**: `hf auth login` defaults to read access. For write/upload,
   the token needs `write` scope — direct the user to
   https://huggingface.co/settings/tokens if they hit `403 Forbidden`.
6. **HF Jobs (compute)** runs in HF's infrastructure with paid GPUs. Use
   carefully — confirm `--flavor` and `--timeout` with the user before
   launching a long job.

## Resources

- [CLI Reference](https://huggingface.co/docs/huggingface_hub/guides/cli) — complete command docs
- [Token Settings](https://huggingface.co/settings/tokens) — manage tokens / scopes
- [Jobs Documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) — paid compute jobs
- [Installing the agent skill](https://agentskills.io) — the auto-generated, always-current skill file (run `hf skills add --claude --global` for Claude Code)

## When NOT to use this skill

- Hub website navigation / UI questions — those are not CLI-related.
- Programmatic access from Python: `huggingface_hub` SDK is a separate skill territory.
- Model inference (calling a hosted model) — that's the Inference API / `text-generation-inference`, not the CLI.

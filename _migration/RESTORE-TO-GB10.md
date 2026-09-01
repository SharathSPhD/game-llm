# Restore 5090 → new GB10 (when the RMA replacement arrives)

This reverses the 2026-09-01 GB10→5090 migration. Everything you need lives on
the 5090 at `/home/ss/.claude-gb10/` (sessions + config + these scripts) and
`/home/ss/projects/` (project files). The reverse script was **validated by
dry-run** on 2026-09-01 — it round-trips cleanly.

## Before you start

- The new GB10 will be a **fresh machine**. Decide its username. If you recreate
  the **same** user `sharaths`, sessions restore to their ORIGINAL path-identical
  layout (`/home/sharaths`) and resume with zero surprises. The script below
  takes the username as an argument, so any name works.
- **Work done on the 5090 during the RMA window is included automatically** —
  the restore reads the *current* state of `/home/ss/.claude-gb10`, not a replay
  of the original migration. New sessions, new commits, new memory all come back.
- **OAuth never transfers** (same as the forward trip): on the new GB10, run
  `claude` once and `/login` via browser. Everything else is files.

## Step 1 — push any 5090 work to GitHub first (safety, like Phase A was)

On the 5090, for each project you worked on:
```bash
cd /home/ss/projects/<project>
git push --all origin        # or push the specific branches you advanced
```

## Step 2 — produce the restored session tree (run ON THE 5090)

```bash
# <USER> = the new GB10's username (e.g. sharaths)
python3 /home/ss/.claude-gb10/restore_to_gb10.py <USER> \
        /home/ss/.claude-gb10/projects \
        /tmp/restore-to-gb10/projects
```
Expect: `dirs=30 files_rewritten=~1535`, `residual /home/ss files: 0`.

## Step 3 — copy everything to the new GB10

```bash
NEWGB10=<user>@<new-gb10-ip>

# 3a. project files (adjust excludes as you like; venvs/node_modules regenerate)
rsync -a --info=progress2 \
  --exclude='.venv/' --exclude='node_modules/' --exclude='__pycache__/' \
  --exclude='.mypy_cache/' --exclude='.ruff_cache/' --exclude='.pytest_cache/' \
  /home/ss/projects/  $NEWGB10:/home/<USER>/projects/

# 3b. restored Claude sessions
rsync -a /tmp/restore-to-gb10/projects/  $NEWGB10:/home/<USER>/.claude/projects/

# 3c. config: rewrite .claude.json paths /home/ss -> /home/<USER>, then copy
sed -e 's#/home/ss/projects#/home/<USER>/projects#g' \
    -e 's#/home/ss/.claude-gb10#/home/<USER>/.claude#g' \
    -e 's#/home/ss#/home/<USER>#g' \
    /home/ss/.claude-gb10/.claude.json  > /tmp/claude.json.new
rsync -a /tmp/claude.json.new  $NEWGB10:/home/<USER>/.claude.json
rsync -a /home/ss/.claude-gb10/settings.json  $NEWGB10:/home/<USER>/.claude/settings.json
```

## Step 4 — first run on the new GB10

```bash
ssh $NEWGB10
export PATH="$HOME/.local/bin:$PATH"     # if claude isn't already on PATH
# install Claude Code if the fresh box doesn't have it:
#   curl -fsSL https://claude.ai/install.sh | bash
cd ~/projects/game-llm
claude          # first run: /login (browser OAuth)
claude --resume # your migrated + RMA-window sessions all listed
```
If you recreated user `sharaths`, you do NOT need CLAUDE_CONFIG_DIR — sessions
live at the default `~/.claude`. (Only the 5090 needed CLAUDE_CONFIG_DIR because
its config dir was a non-default `.claude-gb10` to avoid clobbering ss's own.)

## Step 5 — smoke test (same as the forward trip)

```bash
cd ~/projects/game-llm
sid=$(ls -t ~/.claude/projects/-home-<USER>-projects-game-llm/*.jsonl | head -1 | xargs basename | sed s/.jsonl//)
claude -p --resume "$sid" "What project is this and what was the last topic in this session?"
```
Expect it to recall real prior context. Then rebuild a venv and run that
project's tests to confirm the environment.

## Step 6 — repopulate weights left behind

The heavy checkpoints were NOT migrated (they stayed on the RMA'd unit / went to
HF). Re-pull what you need:
- prabhasa HORD: `huggingface-cli download qbz506/p-s-hord-m2` (and `-m3`)
- Anything else: from its GitHub remote or HF repo (see `MIGRATION.md`).

## Files that make this work (all on the 5090)

- `/home/ss/.claude-gb10/restore_to_gb10.py` — the reverse script (validated)
- `/home/ss/.claude-gb10/projects/` — all migrated + RMA-window sessions
- `/home/ss/.claude-gb10/.claude.json`, `settings.json` — config
- `/home/ss/projects/` — project files
- `/home/ss/projects/MIGRATION.md` — the forward-trip record (context)
- `/home/ss/gb10-rma-backup/dotclaude/` — untouched byte copy of the original
  GB10 `~/.claude` (fallback if anything above is questioned)

## Work-traces (conversation history as readable Markdown)

312 substantial session transcripts were exported to Markdown + index and kept
PRIVATE (never pushed — 3/4 source repos are public). Location:
- 5090: `/home/ss/gb10-work-traces/<project>/` (game-llm, MIabstraction,
  prabhasa-samskrutam, vakya-vallari) + `export_traces.py` to regenerate.
- Restore to new GB10: `rsync -a /home/ss/gb10-work-traces/ NEWGB10:~/gb10-work-traces/`
- `session-traces/` is gitignored in those repos so re-exports never commit.
- Raw `.jsonl` transcripts also live in `~/.claude/projects/` (CLI-resumable).

## SECURITY — before the RMA GB10 ships (added)

- [ ] **Rotate the GitHub PAT** (`ghp_ertf…`) found in a prabhasa transcript —
      revoke at github.com/settings/tokens (user actioned 2026-09-01).
- [ ] Wipe `~/.claude.json` RunPod API key (`rpa_…`) + rotate.
- [ ] The work-traces contain secrets — keep them PRIVATE (never push to the
      public repos game-llm / MIabstraction / vakya-vallari).

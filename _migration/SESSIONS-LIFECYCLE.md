# Chat & cowork sessions — preserve / use on 5090 / recover to new GB10

Definitive guide to your Claude session history across the RMA move.
Investigated from the actual session files on 2026-09-01.

## The two kinds of session on this machine (882 total)

| Kind | Count | How stored | Tied to |
|------|-------|-----------|---------|
| **Cowork / Claude Desktop** (`type: bridge-session`, `entrypoint: claude-desktop`) | **524** | local `.jsonl` transcript **+** a cloud `bridgeSessionId` (`cse_…`) | your **Claude account** (`ownerAccountUuid 92662007…`) **and** the machine |
| **Claude Code CLI** (local) | **358** | local `.jsonl` only | the **machine** only |

Key fact: **cowork sessions are dual** — the *transcript* lives locally
(`~/.claude/projects/<encoded-folder>/<id>.jsonl`), but the *session identity*
is registered in your Claude account on Anthropic's servers. So the account
half survives machine loss automatically; the transcript half must be moved as
a file.

(Separately, any **pure Claude Desktop chats** that are NOT connected to a
remote folder live entirely in your claude.ai account — not on this machine at
all — and need no migration.)

## PRESERVE (done — three independent layers)

1. **Raw `.jsonl` transcripts** — all 882 migrated to the 5090
   (`~/.claude/projects/` merged into the default config, and a copy in
   `~/.claude-gb10/`). This is the full-fidelity record; cowork + CLI both.
2. **Cloud account references** — the 524 cowork sessions' `bridgeSessionId`s
   are on Anthropic's servers under your account; nothing to do, they persist.
3. **Readable Markdown export** — 312 substantial sessions rendered to `.md`
   + index in `~/gb10-work-traces/` (private, gitignored). This is the
   **most durable** layer: independent of `.jsonl` format, of the
   `bridgeSessionId` association, and of any app UI. If everything else broke,
   the Markdown is still a searchable human record of the work.

## USE on the 5090 (during the RMA window)

- Point **Claude Desktop → the 5090** (`ss@192.168.0.204`); open folders under
  `/home/ss/projects/...`. New cowork sessions save their transcript locally on
  the 5090 **and** register with your account (verified: the live `testing`
  session wrote `type: bridge-session`, `entrypoint: claude-desktop`).
- Old transcripts are present on the 5090. Note Desktop's composer starts a new
  session from a **git branch**, so it will not replay an old transcript inline;
  to read/continue a specific old one:
  - open the readable Markdown in `~/gb10-work-traces/<project>/` (fastest), or
  - `CLAUDE_CONFIG_DIR=/home/ss/.claude claude --resume <session-id>` in a
    terminal on the 5090 (works for any of the 882; verified).
- The `bridgeSessionId` cloud references mean your account still "knows" the
  cowork sessions regardless of machine.

## RECOVER to the new GB10 (when the RMA unit arrives)

1. **Transcripts** — run `restore_to_gb10.py` (reverse-migration, validated) to
   rewrite the session dirs/paths from `/home/ss` back to the new user's home,
   then rsync into `~/.claude/projects/`. This restores all 882, cowork + CLI.
2. **Account references** — automatic. `/login` on the new GB10 with the same
   account; the 524 cowork sessions' cloud identities are already there.
3. **Markdown traces** — `rsync -a ~/gb10-work-traces/ NEWGB10:~/gb10-work-traces/`.
4. **Reconnect Desktop** — point Claude Desktop at the new GB10; open the
   restored folders. New cowork sessions resume the local+cloud dual pattern.

## Why nothing is lost

- Cowork sessions: transcript (file, migrated) + identity (account, persistent)
  + Markdown (durable export). Triple-covered.
- CLI sessions: transcript (file, migrated) + Markdown for the substantial ones.
- The Markdown export is the machine- and app-independent safety net.

## Note on `gb10-archive` (dormant projects)

`~/gb10-archive` on the 5090 (85G, 25 dormant projects — dreamprice, slm-1,
gptbert_v04_*, pwm-phase*, etc.) is **already the sole copy** of those projects
(they are 8–16K stubs on the GB10, archived+deleted 2026-08-26). It is **not**
a duplicate of the `projects/` migration and does **not** need re-copying to the
new GB10 — leave it on the 5090. Only pull a specific dormant project back from
`gb10-archive` if you decide to reactivate it.

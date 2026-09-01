# Claude Desktop continuity + skills/plugins/connectors restore

Companion to `MIGRATION.md` / `RESTORE-TO-GB10.md`. Covers the Mac-Mini
Claude Desktop setup and how skills, plugins, and MCP connectors move.

## The architecture (why sessions were at risk)

```
Mac Mini (Claude Desktop GUI)  --network-->  GB10 (Claude Code remote-server, ~/.claude/remote/ccd-cli)
                                             +-- sessions, files, compute live HERE
```

Claude Desktop on the Mac is a **thin client**. When you open a GB10 folder,
Claude Code actually runs **on the GB10**, and the cowork/code sessions are
stored on the GB10 at `~/.claude/projects/*.jsonl`. So yes — those sessions
would die with the GB10. That is exactly what this migration backs up.

**Claude Desktop and Claude Code CLI are the same session store here** (Desktop
drives Claude Code over the remote bridge), so the migrated `.jsonl` sessions
ARE the Desktop sessions. There is no separate Desktop copy to worry about.

## During the RMA window — point Claude Desktop at the 5090

The migrated sessions were merged into the 5090's **default** config
`/home/ss/.claude/` (not just `.claude-gb10`), precisely so Desktop sees them.
Verified: resuming a migrated session from the 5090 default config recalls
prior context.

To reconnect:
1. In Claude Desktop (Mac Mini), add/point the remote host to the **5090**:
   `ss@192.168.0.204` (same LAN, SSH key already works from the GB10; add the
   Mac's key to the 5090 `~/.ssh/authorized_keys` if not already).
2. On first connect Desktop installs its remote bridge on the 5090
   (`/home/ss/.claude/remote/` — it did not exist yet; created automatically).
3. Open folders under `/home/ss/projects/...`. The encoded session dirs
   (`-home-ss-projects-<name>`) match, so your sessions appear in the picker.
4. First run needs `/login` (OAuth never transfers) — already done on 5090.

What's already in place on the 5090 default config (`/home/ss/.claude`):
- 214 game-llm sessions + all other project histories (merged)
- 104 skills (incl. the 3 user-authored: academic-paper-style,
  efe-autoresearch, rtx5090-connect)
- 2.3G plugins tree + 8 marketplaces + 103 enabledPlugins (paths rewritten)
- `.claude.json` merged: ss's fresh login + migrated project history + runpod MCP
- `settings.json` merged: enabledPlugins + marketplaces

## Skills / plugins / connectors — how they restore

**Skills** (`~/.claude/skills/`, 104 dirs, 17M): most come FROM plugins and
return when plugins reinstall. **3 are user-authored and exist nowhere else** —
they MUST be copied: `academic-paper-style`, `efe-autoresearch`,
`rtx5090-connect`. (Already copied to the 5090.)

**Plugins** (`~/.claude/plugins/`, 2.3G): two valid restore paths —
- *Reinstall* (clean, gets updates): source of truth is
  `known_marketplaces.json` (8 marketplaces) + `settings.json` `enabledPlugins`.
  On a fresh box, `claude` re-fetches them from their git repos.
- *Byte-copy* (offline, exact versions): copy the whole `plugins/` tree, then
  rewrite embedded paths `sed -i 's#/home/<old>#/home/<new>#g'` in
  `installed_plugins.json` and `known_marketplaces.json` (they hardcode
  `installLocation` paths). This is what was done GB10->5090.

**Connectors (MCP)**: defined in `.claude.json` `mcpServers` (+ per-project).
- The `runpod` server carries an inline **API key in cleartext** — it travels
  with `.claude.json`. **Rotate it and/or wipe it from the RMA unit before
  shipping** (see pre-ship checklist in MIGRATION.md).
- OAuth-based connectors (Supabase, Figma, GitHub, Slack, etc.) store **no
  reusable token** — they must be **re-authorized via `/mcp`** on each new
  machine, exactly like `/login`.

## Restoring all this to the NEW GB10

`RESTORE-TO-GB10.md` handles projects + sessions. Additionally:
```bash
# skills (user-authored ones are the must-haves; copy all for convenience)
rsync -a /home/ss/.claude/skills/  NEWGB10:/home/<user>/.claude/skills/
# plugins: either let `claude` reinstall from marketplaces, OR byte-copy:
rsync -a /home/ss/.claude/plugins/ NEWGB10:/home/<user>/.claude/plugins/
#   then on NEWGB10: sed -i 's#/home/ss#/home/<user>#g' \
#     ~/.claude/plugins/installed_plugins.json ~/.claude/plugins/known_marketplaces.json
# settings (enabledPlugins/marketplaces) travel in settings.json (RESTORE step 3c)
# connectors: re-auth OAuth ones via /mcp; re-add runpod key (rotated) to .claude.json
```

## Pre-ship checklist for the RMA GB10 (before it leaves)

- [ ] Wipe secrets: `~/.claude.json` runpod API key, any `.env*`, credentials.
      (Best: full-disk sanitize per NVIDIA guide — UEFI Media Sanitization.)
- [ ] Confirm 5090 has everything (this doc's "already in place" list).
- [ ] Re-enable Secure Boot (disabled for FieldDiag).

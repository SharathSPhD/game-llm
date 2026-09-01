#!/usr/bin/env python3
"""Export Claude Code session transcripts (.jsonl) -> compact readable Markdown,
plus a master index. Durable, git-trackable 'trace of the work'.

Usage: export_traces.py <project_substring> <out_dir> [min_turns]
Produces: <out_dir>/<sessionid>__<date>__<slug>.md  + appends to <out_dir>/INDEX.md
Compact: user prompts + assistant prose in full; tool calls summarized to one
line (name + short arg); large tool RESULTS truncated (they bloat and are
reproducible). The narrative/decisions — the actual trace — is kept.
"""
import os, sys, json, glob, re
from datetime import datetime, timezone

SRC = "/home/sharaths/.claude/projects"
PROJ = sys.argv[1]
OUT = sys.argv[2]
MIN_TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
os.makedirs(OUT, exist_ok=True)


def encoded_dirs(proj):
    key = "-projects-" + proj.lower().replace("_", "-")
    return [d for d in os.listdir(SRC)
            if os.path.isdir(os.path.join(SRC, d)) and key in d.lower()]


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                out.append(b.get("text", ""))
            elif t == "tool_use":
                arg = json.dumps(b.get("input", {}))[:120]
                out.append(f"\n> 🔧 **{b.get('name')}**(`{arg}`)")
            elif t == "tool_result":
                r = b.get("content", "")
                if isinstance(r, list):
                    r = " ".join(x.get("text", "") for x in r if isinstance(x, dict))
                r = str(r).replace("\n", " ")
                out.append(f"\n> ↳ _result: {r[:200]}{'…' if len(r) > 200 else ''}_")
        return "".join(out)
    return ""


def slug(s):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip().lower())[:50].strip("-")
    return s or "session"


def export_file(path, proj):
    turns, first_prompt, ts_first, ts_last = [], None, None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ts = d.get("timestamp")
        if ts:
            ts_first = ts_first or ts
            ts_last = ts
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        body = text_of(m.get("content", "")).strip()
        if not body:
            continue
        if role == "user" and first_prompt is None and not body.startswith(">"):
            first_prompt = body[:100]
        turns.append((role, body))
    if len([t for t in turns if not t[1].startswith(">")]) < MIN_TURNS:
        return None
    sid = os.path.basename(path)[:-6]
    date = (ts_first or "")[:10] or "nodate"
    fn = f"{date}__{sid[:8]}__{slug(first_prompt or 'session')}.md"
    with open(os.path.join(OUT, fn), "w", encoding="utf-8") as fo:
        fo.write(f"# {first_prompt or sid}\n\n")
        fo.write(f"- Project: {proj}\n- Session: `{sid}`\n")
        fo.write(f"- Start: {ts_first}  End: {ts_last}\n- Turns: {len(turns)}\n\n---\n\n")
        for role, body in turns:
            fo.write(f"### {'🧑 User' if role == 'user' else '🤖 Claude'}\n\n{body}\n\n")
    return {"file": fn, "sid": sid, "date": date, "turns": len(turns),
            "prompt": (first_prompt or "").replace("|", "/")}


def main():
    rows = []
    for d in encoded_dirs(PROJ):
        for f in glob.glob(os.path.join(SRC, d, "*.jsonl")):
            r = export_file(f, PROJ)
            if r:
                rows.append(r)
    rows.sort(key=lambda r: r["date"])
    idx = os.path.join(OUT, "INDEX.md")
    with open(idx, "w", encoding="utf-8") as fo:
        fo.write(f"# Session traces — {PROJ}\n\n{len(rows)} sessions exported.\n\n")
        fo.write("| Date | Turns | Session | First prompt | File |\n|---|---|---|---|---|\n")
        for r in rows:
            fo.write(f"| {r['date']} | {r['turns']} | `{r['sid'][:8]}` | {r['prompt'][:70]} | [{r['file']}]({r['file']}) |\n")
    # machine-readable index
    with open(os.path.join(OUT, "index.csv"), "w", encoding="utf-8") as fo:
        fo.write("date,turns,session_id,first_prompt,file\n")
        for r in rows:
            fo.write(f"{r['date']},{r['turns']},{r['sid']},\"{r['prompt']}\",{r['file']}\n")
    print(f"{PROJ}: exported {len(rows)} traces -> {OUT}")


if __name__ == "__main__":
    main()

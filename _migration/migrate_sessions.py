#!/usr/bin/env python3
"""Phase D: migrate Claude Code sessions from GB10 (/home/sharaths) layout to
5090 (/home/ss) layout.

Two transforms, applied to a COPY (originals on GB10 are never touched):
  1. Rename each path-encoded project dir:
       -home-sharaths-projects-X  ->  -home-ss-projects-X
  2. Rewrite embedded absolute paths inside every *.jsonl line:
       /home/sharaths/projects  ->  /home/ss/projects
       /home/sharaths/.claude   ->  /home/ss/.claude-gb10   (config dir)

Run locally on GB10 to PRODUCE the migrated tree under an output dir, which is
then rsynced to the 5090. Idempotent; safe to re-run.
"""
import os, sys, shutil, re

SRC = "/home/sharaths/.claude/projects"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-1000/claude-migrated/projects"

OLD_HOME = "/home/sharaths"
NEW_HOME = "/home/ss"
# project path segment: keep identical relative layout under new home
SUBS = [
    ("/home/sharaths/projects", "/home/ss/projects"),
    ("/home/sharaths/.claude", "/home/ss/.claude-gb10"),
    ("/home/sharaths", "/home/ss"),  # catch-all last (e.g. bare cwd refs)
]

def new_dirname(name):
    # encoded dir names look like -home-sharaths-projects-...
    return name.replace("-home-sharaths-", "-home-ss-", 1)

def rewrite_line(line):
    for old, new in SUBS:
        if old in line:
            line = line.replace(old, new)
    return line

def main():
    os.makedirs(OUT, exist_ok=True)
    n_dirs = n_files = n_lines = 0
    for d in sorted(os.listdir(SRC)):
        src_dir = os.path.join(SRC, d)
        if not os.path.isdir(src_dir):
            continue
        dst_name = new_dirname(d)
        dst_dir = os.path.join(OUT, dst_name)
        os.makedirs(dst_dir, exist_ok=True)
        n_dirs += 1
        # walk the whole session dir tree; rewrite text-ish files, copy the rest
        for root, _dirs, fnames in os.walk(src_dir):
            rel = os.path.relpath(root, src_dir)
            out_root = os.path.join(dst_dir, rel) if rel != "." else dst_dir
            os.makedirs(out_root, exist_ok=True)
            for fn in fnames:
                sp = os.path.join(root, fn)
                dp = os.path.join(out_root, fn)
                if fn.endswith((".jsonl", ".json", ".txt", ".md", ".js", ".sh", ".py")):
                    try:
                        with open(sp, encoding="utf-8") as fi:
                            data = fi.read()
                    except (UnicodeDecodeError, OSError):
                        shutil.copy2(sp, dp)  # binary/unreadable: copy as-is
                        continue
                    out = data
                    for old, new in SUBS:
                        out = out.replace(old, new)
                    if out != data:
                        n_lines += out.count(NEW_HOME) - data.count(NEW_HOME)
                    with open(dp, "w", encoding="utf-8") as fo:
                        fo.write(out)
                    n_files += 1
                else:
                    shutil.copy2(sp, dp)
    print(f"dirs={n_dirs} jsonl_files={n_files} rewritten_lines={n_lines}")
    print(f"output: {OUT}")

if __name__ == "__main__":
    main()

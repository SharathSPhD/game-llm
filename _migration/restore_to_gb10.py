#!/usr/bin/env python3
"""REVERSE migration: 5090 (/home/ss) -> new GB10 when the RMA unit arrives.

Inverse of migrate_sessions.py. Rewrites Claude session dirs + embedded paths
from the 5090's /home/ss layout back to the new GB10's home layout.

Run this ON THE 5090, producing a migrated tree, then rsync it to the new GB10.
Originals on the 5090 are never touched (writes to a separate OUT dir).

Usage:
    python3 restore_to_gb10.py <NEW_GB10_USER> [SRC_CONFIG_DIR] [OUT_DIR]

  NEW_GB10_USER   the username on the new GB10 (e.g. "sharaths").
                  If the new box uses the SAME user as the old GB10 (sharaths),
                  this restores the ORIGINAL path-identical layout and the
                  sessions resume with zero further edits.
  SRC_CONFIG_DIR  default /home/ss/.claude-gb10  (the migrated config on 5090)
  OUT_DIR         default /tmp/restore-to-gb10/projects

After producing OUT_DIR, on the 5090:
    rsync -a /tmp/restore-to-gb10/projects/  NEWGB10:/home/<user>/.claude/projects/
    rsync -a /home/ss/.claude-gb10/.claude.json  NEWGB10:/home/<user>/.claude.json  # then edit paths (below)
    rsync -a /home/ss/projects/  NEWGB10:/home/<user>/projects/   # the actual project files
And on the new GB10: run `claude` once and `/login` (OAuth never transfers).
"""
import os, sys, shutil

USER = sys.argv[1] if len(sys.argv) > 1 else "sharaths"
SRC = sys.argv[2] if len(sys.argv) > 2 else "/home/ss/.claude-gb10/projects"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/restore-to-gb10/projects"

NEW_HOME = f"/home/{USER}"
# Reverse of the forward transform. Order matters: most specific first.
SUBS = [
    ("/home/ss/projects", f"{NEW_HOME}/projects"),
    ("/home/ss/.claude-gb10", f"{NEW_HOME}/.claude"),
    ("/home/ss", NEW_HOME),
]
REWRITE_EXT = (".jsonl", ".json", ".txt", ".md", ".js", ".sh", ".py")


def new_dirname(name):
    # -home-ss-projects-X  ->  -home-<user>-projects-X
    return name.replace("-home-ss-", f"-home-{USER}-", 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    n_dirs = n_files = 0
    for d in sorted(os.listdir(SRC)):
        src_dir = os.path.join(SRC, d)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(OUT, new_dirname(d))
        os.makedirs(dst_dir, exist_ok=True)
        n_dirs += 1
        for root, _dirs, fnames in os.walk(src_dir):
            rel = os.path.relpath(root, src_dir)
            out_root = os.path.join(dst_dir, rel) if rel != "." else dst_dir
            os.makedirs(out_root, exist_ok=True)
            for fn in fnames:
                sp, dp = os.path.join(root, fn), os.path.join(out_root, fn)
                if fn.endswith(REWRITE_EXT):
                    try:
                        data = open(sp, encoding="utf-8").read()
                    except (UnicodeDecodeError, OSError):
                        shutil.copy2(sp, dp); continue
                    out = data
                    for old, new in SUBS:
                        out = out.replace(old, new)
                    open(dp, "w", encoding="utf-8").write(out)
                    n_files += 1
                else:
                    shutil.copy2(sp, dp)
    print(f"dirs={n_dirs} files_rewritten={n_files}")
    print(f"output: {OUT}")
    print(f"target user: {USER}  ->  {NEW_HOME}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# check_style.sh — mechanical style checks for an academic LaTeX manuscript.
#
# Usage: check_style.sh <paper-dir> [main-tex-basename]
#
# Reports violations; fixes nothing. Judgement calls (does this section argue or
# merely summarise?) still need a human read. Exit status is the violation count,
# so it can gate a commit hook.

set -uo pipefail
DIR="${1:-.}"
MAIN="${2:-}"
SECTIONS="$DIR/sections"
[ -d "$SECTIONS" ] || SECTIONS="$DIR"
FAIL=0

hdr() { printf '\n\033[1m%s\033[0m\n' "$1"; }
bad() { printf '  \033[31m%s\033[0m\n' "$1"; FAIL=$((FAIL+1)); }
ok()  { printf '  \033[32m%s\033[0m\n' "$1"; }

hdr "First-person voice (should be third-person impersonal)"
HITS=$(grep -rniE '\b(we|our|ours|us)\b' "$SECTIONS"/*.tex 2>/dev/null \
       | grep -viE 'cite|url|http|%' || true)
if [ -n "$HITS" ]; then
  echo "$HITS" | head -12 | while IFS= read -r l; do bad "${l:0:150}"; done
  bad "total: $(echo "$HITS" | wc -l) first-person occurrences"
else ok "none"; fi

hdr "List environments (body prose should carry these as prose or tables)"
HITS=$(grep -rn 'begin{itemize}\|begin{enumerate}' "$SECTIONS"/*.tex 2>/dev/null || true)
if [ -n "$HITS" ]; then
  echo "$HITS" | while IFS= read -r l; do bad "${l:0:150}"; done
else ok "none"; fi

hdr "Inline enumerations (lists with the bullets filed off)"
HITS=$(grep -rnE '(: |; )\(1\) |\(2\) [A-Z]|\(3\) [A-Z]' "$SECTIONS"/*.tex 2>/dev/null || true)
if [ -n "$HITS" ]; then
  echo "$HITS" | head -8 | while IFS= read -r l; do bad "${l:0:150}"; done
else ok "none"; fi

hdr "Vocabulary that signals machine drafting"
HITS=$(grep -rniE 'honest|adversarially audited|commend|notably,|importantly,|crucially|delve|showcase|it is worth noting' \
       "$SECTIONS"/*.tex 2>/dev/null || true)
if [ -n "$HITS" ]; then
  echo "$HITS" | head -10 | while IFS= read -r l; do bad "${l:0:150}"; done
else ok "none"; fi

hdr "Figures generated but never included"
if [ -d "$DIR/figures" ]; then
  for f in "$DIR"/figures/*.pdf "$DIR"/figures/*.png; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); stem="${b%.*}"
    grep -rq "$stem" "$SECTIONS"/*.tex 2>/dev/null || bad "unused figure: $b"
  done
  ok "checked $(ls "$DIR"/figures/*.pdf "$DIR"/figures/*.png 2>/dev/null | wc -l) figure files"
else ok "no figures/ directory"; fi

hdr "Figures included but never referenced from prose"
for lab in $(grep -rho '\\label{fig:[^}]*}' "$SECTIONS"/*.tex 2>/dev/null | sed 's/.*{\(.*\)}/\1/' | sort -u); do
  grep -rq "ref{$lab}" "$SECTIONS"/*.tex 2>/dev/null || bad "figure never referenced: $lab"
done
ok "reference check complete"

hdr "Duplicate labels"
# Skip backup/draft copies: a stale duplicate of the manuscript in the same
# directory would otherwise report every label as doubly declared.
DUPES=$(grep -rho '\\label{[^}]*}' "$SECTIONS"/*.tex "$DIR"/*.tex 2>/dev/null \
        --exclude='*backup*' --exclude='*draft*' --exclude='*-old*' \
        | sort | uniq -d || true)
if [ -n "$DUPES" ]; then
  echo "$DUPES" | while IFS= read -r l; do bad "declared more than once: $l"; done
else ok "none"; fi

hdr "Proportions"
if [ -d "$DIR/sections" ]; then
  TOTAL=$(cat "$SECTIONS"/*.tex 2>/dev/null | wc -w)
  RES=$(wc -w < "$SECTIONS/results.tex" 2>/dev/null || echo 0)
  printf '  section prose: %s words (reference manuscript: ~15000)\n' "$TOTAL"
  printf '  results: %s words\n' "$RES"
  BIGGEST=$(wc -w "$SECTIONS"/*.tex 2>/dev/null | sort -rn | sed -n '2p' | awk '{print $2}')
  case "$BIGGEST" in
    *results.tex) ok "results is the longest section" ;;
    *) bad "results is not the longest section (longest: $(basename "${BIGGEST:-unknown}"))" ;;
  esac
  printf '  tables: %s | figures: %s\n' \
    "$(grep -rc 'begin{table}' "$SECTIONS"/*.tex 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')" \
    "$(grep -rc 'begin{figure}' "$SECTIONS"/*.tex 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')"
fi

if [ -n "$MAIN" ] && command -v latexmk >/dev/null 2>&1; then
  hdr "Build"
  if (cd "$DIR" && timeout 300 latexmk -pdf -interaction=nonstopmode -halt-on-error "$MAIN" >/dev/null 2>&1); then
    LOG="$DIR/${MAIN%.tex}.log"
    ok "builds; $(pdfinfo "$DIR/${MAIN%.tex}.pdf" 2>/dev/null | awk '/Pages/{print $2}') pages"
    UC=$(grep -ci 'undefined citation' "$LOG" 2>/dev/null | head -1)
    UC=${UC:-0}
    [ "$UC" -gt 0 ] && bad "undefined citations: $UC" || ok "no undefined citations"
  else bad "build failed"; fi
fi

hdr "Result"
if [ "$FAIL" -eq 0 ]; then ok "no mechanical violations"; else printf '  \033[31m%s issue(s) to address\033[0m\n' "$FAIL"; fi
exit "$FAIL"

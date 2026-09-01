---
name: academic-paper-style
description: Publication-grade academic writing standard — third-person impersonal voice, no bullet lists in the body, dense booktabs tables and TikZ figures, modular LaTeX sections, and vocabulary that avoids AI-generated tells. Use this whenever writing or revising a paper, manuscript, journal or conference submission, abstract, results/methods/discussion section, related work, appendix, or a LaTeX document intended for publication — and also when asked to "improve", "deepen", "tighten" or "make publication quality" any academic prose, even if the word "paper" is never used. Also applies when updating a paper after new experimental results land.
---

# Academic paper style

Papers drafted by a model tend to fail in the same recognisable ways: they bullet
what should be argued, narrate the research process instead of reporting findings,
lapse into promotional register, and stay thin where a reviewer wants depth. This
skill encodes a standard that avoids those failures. It was derived by measuring a
real MDPI journal submission that passed peer review, so the targets below are
observed practice rather than preference.

## The measured benchmark

The reference manuscript contains roughly 15,000 words of section prose, twelve
figures, twelve tables, and **zero** `itemize` or `enumerate` environments in the
entire body. Its results section is more than three times the length of any other
section. Prose lives in `sections/*.tex`, each file opening with a comment recording
which run directory its numbers came from; the root `.tex` holds only preamble,
metadata and `\input` lines.

Use those proportions as a sanity check. A results section shorter than the
background section usually means the analysis was summarised rather than argued.

## Voice

Write in third-person impersonal. "Retention is measured against the base model
under an identical harness invocation." Not "we measure retention". The first
person creeps back in during revision, so it is worth re-grepping before declaring
a draft finished.

This is not merely a convention. Impersonal phrasing forces the sentence to name
the thing that acts — the protocol, the model, the measurement — which tends to
expose vagueness that "we tried X" conceals.

## No lists in the body

Contributions, experimental arms, limitations, and gates all belong in prose or in
a table. If a passage resists being written as prose, that is usually a signal the
relationships between the items were never worked out; a list lets the author skip
that work, and a reader notices.

Watch for lists in disguise: a paragraph reading "Three consequences follow: (1)…
(2)… (3)…" is a bulleted list with the bullets filed off. Rewrite it so each claim
connects to the next.

## Depth over summary

Every experimental claim carries four things: the protocol that produced it, the
number itself, what the number means, and the alternative explanation that was
excluded. The last of these is what separates a results section from a table
caption, and it belongs inside the subsection for that experiment rather than
collected into a separate "alternatives considered" block.

State scope limits as plain technical constraints in a Limitations section —
sample sizes, confounds, what the metric does and does not measure. Do not
editorialise about the virtue of disclosing them.

## Vocabulary that signals machine drafting

Remove on sight: *honest*, *honestly*, *adversarially audited*, *commend*,
*Notably,*, *Importantly,*, *Crucially,*, *delve*, *showcase*, *it is worth
noting*, *robust* used as filler, *leverage* as a verb meaning "use". Also avoid
stacking rhetorical em-dash asides; at most one dash construction per paragraph.

Self-congratulation about research integrity is the most persistent tell. A
sentence such as "the negative results are reported honestly" should become a
statement of what the results were and how they were scored. Reporting a miss at
the same length and evidentiary standard as a pass demonstrates the point without
announcing it.

## Apparatus

Tables use `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`), `\multirow`
where rows group under a model or condition, and confidence intervals wherever the
statistic supports them. Captions state the protocol — seeds, sample counts, what
is held fixed — not just a label.

Figures should include architecture and flow diagrams drawn in TikZ, not only
plotted results. A diagram that shows the mechanism carries argument that a
loss curve cannot. Every generated plot that exists should be placed in the
section that discusses its evidence and referenced from the prose; unreferenced
figures are a common oversight when a draft is assembled in pieces.

See `references/latex-patterns.md` for ready table, TikZ and figure skeletons.

## Source layout

Keep the root file to preamble, metadata and `\input` lines, with prose in
`sections/`. Declare each `\label` exactly once — a label in both the root file and
the section file produces multiply-defined warnings that are easy to miss.

Open each section file with a comment naming the run directories or finding
identifiers its numbers come from. When a reviewer asks where a figure came from
eighteen months later, that line is the answer.

## Numbers

Every number must trace to a results file, a findings ledger, or a cited source
before it enters the manuscript. A derived quantity (a ratio computed from two
measured values) needs its derivation shown or its inputs reported instead —
presenting it bare makes it indistinguishable from an invented one.

Citations generated from memory are unreliable in a specific way: the title and
venue are often right while the author list is fabricated. Verify author lists
against the actual paper before the bibliography is considered done.

## Keeping the paper current

A paper is a living artifact alongside the code. When an experiment closes, its
section, tables, figures and any companion site or application surface update in
the same cycle — not as a later cleanup pass. Deferred writing is where provenance
gets lost and where numbers drift from the runs that produced them.

## Before declaring a draft finished

Run `scripts/check_style.sh <paper-dir>` for the mechanical checks: first-person
usage, list environments, banned vocabulary, inline enumerations, unreferenced
figures, duplicate labels, and a build with citation warnings surfaced. It reports
what to fix rather than fixing it. Judgement calls — whether a section argues or
merely summarises — still need a read-through.

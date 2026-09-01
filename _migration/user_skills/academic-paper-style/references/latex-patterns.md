# LaTeX patterns

Skeletons for the apparatus the standard expects. Copy and adapt rather than
inventing structure per document — consistency across tables is itself a quality
signal to reviewers.

## Contents

- Grouped results table (booktabs + multirow + CIs)
- Comparison table for related approaches
- Per-seed appendix table
- Architecture diagram (TikZ)
- Process/flow diagram (TikZ)
- Figure inclusion with a protocol caption
- Preamble packages

## Grouped results table

Use `\multirow` when several methods are compared under each model or condition.
Confidence intervals go in their own column rather than inline in parentheses,
which keeps the numeric columns scannable.

```latex
\begin{table}[t]
\centering
\caption{Feature discovery under the fair ablation-only protocol: per-prompt mean,
median, inter-quartile range, and bounded oracle efficiency with bootstrap 95\%
confidence intervals over five prompts.}
\label{tab:main}
{\small
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Method} & \textbf{Mean} & \textbf{Median} & \textbf{[Q1, Q3]} & \textbf{Efficiency [95\% CI]} \\
\midrule
\multirow{3}{*}{Model A}
 & Oracle    & 0.000862 & ---      & ---                  & 100.0\% \\
 & Method 1  & 0.000793 & 0.000436 & [0.000399, 0.001306] & 91.9\% [82.1, 96.5] \\
 & Method 2  & 0.000707 & 0.000404 & [0.000335, 0.001115] & 82.0\% [73.8, 87.8] \\
\midrule
\multirow{3}{*}{Model B}
 & Oracle    & 0.009755 & ---      & ---                  & 100.0\% \\
 & Method 1  & 0.009291 & 0.005739 & [0.003297, 0.018357] & 95.2\% [87.0, 97.7] \\
 & Method 2  & 0.008606 & 0.005999 & [0.003302, 0.014895] & 88.2\% [79.0, 96.7] \\
\bottomrule
\end{tabular}
}
\end{table}
```

## Comparison table for related approaches

A background section that contrasts approaches on named axes is more useful than
prose that describes each in turn, and it is the one place a table beats argument.

```latex
\begin{table}[t]
\centering
\caption{Approaches to implicit depth, compared on parameter sharing, the
semantics assigned to depth, and whether a fixed point is certified at
convergence.}
\label{tab:related}
\begin{tabular}{lccc}
\toprule
\textbf{Approach} & \textbf{Parameter sharing} & \textbf{Depth semantics} & \textbf{Certified} \\
\midrule
Explicit stack        & none      & layer index        & n/a \\
Universal Transformer & full      & recurrence step    & no \\
Deep equilibrium      & full      & solver iteration   & at tolerance \\
\bottomrule
\end{tabular}
\end{table}
```

## Per-seed appendix table

Body tables carry aggregates; the appendix carries every seed so a reader can
recompute the aggregate.

```latex
\begin{table}[h]
\centering
\caption{All seeds for the pretraining study. Convergence rate is measured at
evaluation under relative tolerance $10^{-3}$ with a budget of twelve iterations.}
\begin{tabular}{llcccc}
\toprule
\textbf{Arm} & \textbf{Regime} & \textbf{Seed} & \textbf{Score} & \textbf{Conv.} & \textbf{Mean iters} \\
\midrule
A1 & explicit  & 42/43/44 & .682 / .675 / .693 & ---  & ---  \\
A3 & implicit  & 42/43/44 & .537 / .532 / .544 & 0.0  & 12.0 \\
B1 & anytime   & 42/43/44 & .662 / .697 / .672 & 0.0  & 12.0 \\
\bottomrule
\end{tabular}
\end{table}
```

## Architecture diagram

Panels sharing a coordinate scope compare cleanly. Keep node styles in the
`tikzpicture` options so the body stays readable.

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[
  font=\small,
  block/.style={draw, rounded corners=2pt, minimum width=2.1cm, minimum height=0.62cm, align=center},
  explicit/.style={block, fill=black!6},
  tied/.style={block, fill=blue!12, draw=blue!55},
  io/.style={draw, rounded corners=6pt, minimum width=2.1cm, minimum height=0.55cm, fill=black!3},
  ar/.style={-{Latex[length=2mm]}, gray!70},
  lbl/.style={font=\scriptsize, align=center, text=black!65}
]
  \node[io] (in) at (0,0) {input};
  \node[tied, minimum height=1.5cm] (blk) at (0,2.4) {$f_\theta(z,x)$\\\scriptsize tied block};
  \node[io] (out) at (0,4.6) {output};
  \draw[ar] (in) -- (blk);
  \draw[ar] (blk) -- (out);
  \draw[ar, blue!65] (blk.east) .. controls (2.3,2.9) and (2.3,1.9) .. (blk.south east)
    node[lbl, text=blue!65, pos=0.5, right=1pt] {$z \leftarrow f_\theta(z,x)$};
  \node[lbl, anchor=north] at (0,-0.55) {iterated to a fixed point};
\end{tikzpicture}
\caption{Caption states what the diagram shows and what distinguishes the panels,
not merely that it is an architecture.}
\label{fig:arch}
\end{figure}
```

## Process diagram

```latex
\begin{tikzpicture}[
  font=\small, node distance=0.55cm,
  stage/.style={draw, rounded corners=2pt, minimum width=3.1cm, minimum height=0.68cm, align=center, fill=black!4},
  gate/.style={draw, diamond, aspect=2.1, inner sep=1pt, align=center, fill=blue!8, draw=blue!55, font=\scriptsize},
  term/.style={draw, rounded corners=7pt, minimum width=2.4cm, minimum height=0.6cm, align=center, fill=black!3},
  ar/.style={-{Latex[length=2mm]}, gray!75}
]
  \node[term] (a) {start};
  \node[stage, right=1.1cm of a] (b) {stage};
  \node[gate, right=1.1cm of b] (c) {check};
  \node[term, right=1.1cm of c] (d) {outcome};
  \draw[ar] (a) -- (b); \draw[ar] (b) -- (c); \draw[ar] (c) -- (d);
\end{tikzpicture}
```

## Figure inclusion

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.85\textwidth]{figures/fig_name.pdf}
\caption{What is plotted, over how many seeds, under which protocol, and the
comparison the reader should draw. A caption that repeats the axis labels wastes
the most-read line in the paper.}
\label{fig:name}
\end{figure}
```

Reference every figure from the prose with `Figure~\ref{fig:name}`. An included
figure that no sentence points at will be skipped by most readers.

## Preamble

```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, shapes.misc, arrows.meta, positioning, fit,
                decorations.pathreplacing, calc, backgrounds, chains, scopes}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
```

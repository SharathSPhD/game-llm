# 1. Record architecture decisions

Date: 2026-08-20
Status: Accepted

## Context

Kinetic AI is a long research effort (review → science → EqLM → app → paper). Early
decisions — thresholds, baselines, hardware, scope — constrain everything later and
must survive context loss and session restarts.

## Decision

Record significant decisions as ADRs in `docs/decisions/`, numbered sequentially, in
the Nygard format (Context, Decision, Consequences). Pre-registered hypothesis
thresholds (CLAUDE.md H1–H4) change only through a new ADR.

## Consequences

Decision history is auditable in git and forms the MEMORY layer of the closure
contract.

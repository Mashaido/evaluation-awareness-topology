# evaluation-awareness-topology
Detecting evaluation awareness in frontier models through non-linear structures in activation space, where linear probes are known to fail


# Beyond Linearity: Topological Signatures of Evaluation Awareness Under Adversarial Obfuscation

Independent research testing whether non-linear, structural signals in a large language
model's internal representations can detect evaluation awareness, specifically in
the two conditions where current detection methods are known to fail: increased
behavioral complexity, and adversarial training against the detector itself.

**Status:** early stage, infrastructure in progress, no results yet.

**Scope:** phase 1 ((a) Topology-adjacent and (b) Topology) is a detection-only project. Phase 2 addresses intervening on
or correcting evaluation-aware behavior once detected.

## Background

Builds on prior work showing that a model's output can remain fully stable while
its underlying representational structure shifts meaningfully underneath it:

- []Beyond Accuracy: Stable Predictions, Unstable Topology in Transformer Representations](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=q3iC8uAAAAAJ&citation_for_view=q3iC8uAAAAAJ:qjMakFHDy7sC)
- [On the Geometric Limits of Transformer Defenses against Obfuscation Attacks: Latent Embedding Collapse & Performance–Robustness Gap](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=q3iC8uAAAAAJ&citation_for_view=q3iC8uAAAAAJ:2osOgNQ5qMEC)

## Repository structure

- `/data` — matched evaluation/deployment scenario pairs, dataset generation
- `/src` — activation extraction, rank stability metrics, probe baseline
- `/notebooks` — exploratory work
- `/experiments` — run configs
- `/results` — output tables and figures

## Setup

[filled in once environment is finalized]

## Citation

[fill in once published]

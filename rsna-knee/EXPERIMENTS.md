# RSNA Knee — Experiment notes

## Effective or tied with the public baseline

- **R6-w1 + Raptor-384/224 + per-finding blend**: Public **0.958** (reference ensemble)
- **S6 → Raptor-half (`W_S6=0.12`)**: Public 0.958; selected as primary submission
- **Reader @ 0.10**: Public 0.958; selected as secondary submission
- **coat-extra v10geo @ 0.15** (no change to `W_RAPTOR_C`): Public 0.958
- **G233 CoAtNet-320 @ 0.15 + S6**: Public 0.958; lighter arm that still passed hidden runtime

## Ineffective or harmful

- **gold-58 reweight-cons**: offline ≈ +0.009 → Public **0.954**
- Blind single-label edits to `W_RAPTOR` (e.g. Contusion / Fracture): typically **0.957**
- **S6 → compact @ 0.3** (public reproduction path): ≈ **0.957**
- Weak OOF families such as Mycelis: no reliable gain vs the reference ensemble
- Meniscus/Fracture specialist: weaker than the baseline on gold meniscus; deprioritized

## Submission failures (runtime)

Stacks that combined S6 with multiple additional Raptor-384 checkpoints (triple / bs14 / dual-extra / angelus-v4 / fullcohort, etc.) could finish in a public notebook run, but official scoring failed with:

> submission notebook exceeded the allowed runtime

## Private leaderboard

Most Public 0.958 submissions scored Private **0.942**. The final selected seats were not improved by late-stage heavy stacking.

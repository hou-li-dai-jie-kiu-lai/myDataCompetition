# RSNA Knee Abnormality Detection

> 中文版：[README.md](./README.md)

Kaggle · [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · Code Competition

| | |
|---|---|
| **Task** | Multi-label abnormality detection on knee MRI (12 findings) |
| **Metric** | Macro-averaged ROC-AUC |
| **Result** | Private **592 / 6299** (Bronze) · Private **0.942** / Public **0.958** |
| **Team** | Solo |

This folder is a post-competition writeup of the final solution and the main experimental conclusions. It does not redistribute competition data or third-party checkpoints. A compact experiment list is available in [EXPERIMENTS.md](./EXPERIMENTS.md).

---

## Problem setting

The task is to predict twelve clinically relevant findings from a knee MRI study—ligament injury (ACL / MCL), meniscal tear, three-compartment osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion, and fracture. Only a small fraction of training studies carry official gold labels; the remaining studies are typically handled via report-derived weak labels or via publicly released models.

Because this is a **code competition**, every submission is re-run on a hidden test set under a runtime limit. Model quality and inference cost must be optimized jointly.

---

## Final solution

### Selected submissions

Two variants of the same public ensemble were selected for scoring. Both scored Public **0.958** and Private **0.942**.

1. **Primary — S6 mixed into the Raptor branch**  
   Start from the public R6 ⊕ dual-resolution Raptor ensemble with per-finding blend weights. Mix S6 into the **Raptor side** at a fixed weight `W_S6 = 0.12`. If S6 fails, fall back to the ensemble without S6.

2. **Secondary — Reader @ 0.10**  
   Keep the same R6 ⊕ Raptor ensemble and mix in goodpjw’s 2.5D ConvNeXt reader at weight `0.10`.

### Ensemble backbone

```text
R6 ConvNeXt  (Helios fe055406-w1, flip TTA)
        ⊕
Raptor-384 SWA  (nartaa publication CoAtNet)
        ⊕
Raptor-224      (same family; A224 ≈ 0.35)
        ⊕
per-finding blend weights W_RAPTOR_C
```

Public assets used in the final stack:

- `heliosli/rsna-knee-r6-frozen-fe055406-w1`
- `nartaa/rsna-knee-publication-swa-weights-20261007`
- `nartaa/rsna-knee-s6fall-weights`
- `goodpjw2008/rsna-knee-2-5d-convnext-reader` (secondary submission)

Near-cousin CoAtNet weights (e.g. from `manatoshoji03/rsna-knee-coat-extra`) were tested as low-weight additives, but did not improve beyond Public 0.958.

### Inference blend

1. Rank-normalize Raptor-384 and Raptor-224, then combine them with `A224`.  
2. Rank-blend the Raptor branch with R6 using per-finding weights `W_RAPTOR_C`.  
3. For the primary submission, mix S6 ranks into the Raptor branch at `W_S6 = 0.12`.  
4. Limit the number of forward passes so that the hidden-set runtime constraint is satisfied.

---

## Method

### Stage 1 — Reproduce a competitive public ensemble

Self-training and weak-label pipelines initially reached the mid-0.94 range, but lagged behind the strongest public notebooks. The first milestone was therefore a faithful reproduction of a full public stack: R6, dual-resolution Raptor CoAtNet, and per-finding fusion. This yielded a stable Public score of **0.958**, which became the reference point for later work.

### Stage 2 — Search for additive gains under constraints

Subsequent experiments focused on small, controlled modifications that preserve the 0.958 public level and remain within the runtime budget:

- Mix S6 into the Raptor branch rather than into R6/compact (public reproductions of the latter were near 0.957).  
- Add low-weight near-cousin CoAtNet checkpoints without retuning `W_RAPTOR_C`.  
- Prefer lighter input resolutions when testing additional CoAtNet arms.

Directions that failed online or at submission time were discarded:

| Direction | Observation |
|-----------|-------------|
| Retuning blend weights on the 58-study gold set | Public dropped to **0.954**; offline gains did not transfer |
| Stacking multiple additional Raptor-384 models together with S6 | Public notebooks finished, but official submits exceeded the hidden runtime limit |
| Low-quality pseudo-label models | No reliable gain versus the public ensemble |
| Meniscus/Fracture specialist fine-tunes | Weaker than the baseline on gold meniscus labels; unused |

### Stage 3 — Final selection

The public leaderboard contained a large cluster at **0.958**. Private leaders were near **0.960**, while many entries from the 0.958 public band landed around **0.942**. The selected submissions belong to that stable public cluster: they prioritize a faithful strong ensemble and conservative additive components over aggressive late-stage stacking.

---

## Takeaways

1. When strong public ensembles exist, reproducing them is often more valuable than early self-training.  
2. Blend-weight search on a tiny gold subset overfits easily and can hurt the leaderboard.  
3. Hidden-set runtime is part of the objective; a successful public dry-run does not guarantee a scored submission.  
4. Small-weight mixtures of closely related models are usually safer than large changes to the fusion formula.

---

## License

Documentation is released under MIT. Competition data and third-party model weights remain under their original terms and must not be redistributed here.

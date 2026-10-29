# RSNA Knee Abnormality Detection

> English: [README_EN.md](./README_EN.md)

Kaggle · [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · Code Competition

| | |
|---|---|
| **Task** | Multi-label abnormality detection on knee MRI (12 findings) |
| **Metric** | Macro-averaged ROC-AUC |
| **Result** | Private **592 / 6299** (Bronze) · Private **0.942** / Public **0.958** |
| **Team** | Solo |

This repository summarizes the final solution and the main experimental conclusions. Competition data and third-party checkpoints are not redistributed; see [EXPERIMENTS.md](./EXPERIMENTS.md) for a condensed experiment log.

---

## Problem setting

The goal is to predict twelve clinically relevant findings from a knee MRI study, including ligament injury (ACL / MCL), meniscal tear, three-compartment osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion, and fracture. Only a small fraction of training studies provide official gold labels; the remaining studies are typically used through report-derived weak labels or through publicly released models.

The competition is a **code competition**: submissions are re-executed on a hidden test set under a runtime limit. Therefore, solution design must balance predictive quality with inference cost.

---

## Final solution

### Selected submissions

Two complementary variants of the same public ensemble were selected for scoring. Both reached Public **0.958** and Private **0.942**.

1. **Primary — S6 mixed into the Raptor branch**  
   Start from the public R6 ⊕ dual-resolution Raptor ensemble with per-finding blend weights. Mix the S6 ConvNeXt model into the **Raptor side** with a fixed weight `W_S6 = 0.12`. If S6 inference fails, fall back to the ensemble without S6.

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

Public checkpoints used in the final stack include:

- `heliosli/rsna-knee-r6-frozen-fe055406-w1`
- `nartaa/rsna-knee-publication-swa-weights-20261007`
- `nartaa/rsna-knee-s6fall-weights`
- `goodpjw2008/rsna-knee-2-5d-convnext-reader` (secondary submission)

Optional near-cousin CoAtNet weights (e.g. `manatoshoji03/rsna-knee-coat-extra`) were evaluated as low-weight additives, but did not improve beyond Public 0.958.

### Inference blend

1. Rank-normalize Raptor-384 and Raptor-224 predictions, then combine them with coefficient `A224`.  
2. Rank-blend the Raptor branch with R6 using per-finding weights `W_RAPTOR_C`.  
3. For the primary submission, mix S6 ranks into the Raptor branch at `W_S6 = 0.12`.  
4. Keep the number of forward passes small enough to satisfy the hidden-set runtime limit.

---

## Method

### Stage 1 — Reproduce a competitive public ensemble

Early self-training and weak-label pipelines reached the mid-0.94 range, but did not match the strongest public notebooks. The first milestone was therefore to reproduce a full public stack built from R6, dual-resolution Raptor CoAtNet, and per-finding fusion. This established a stable Public score of **0.958**, which served as the reference for subsequent work.

### Stage 2 — Search for additive gains under constraints

Further development focused on small, controlled changes that preserve the 0.958 public level while remaining within the code-competition runtime budget:

- Mix S6 into the Raptor branch rather than into R6/compact (the latter reproduced near 0.957 in public recipes).  
- Add low-weight near-cousin CoAtNet checkpoints without retuning `W_RAPTOR_C`.  
- Prefer lighter resolutions (e.g. 320) when testing additional CoAtNet arms.

Changes that looked promising offline but failed online or in submission were discarded. In particular:

| Direction | Observation |
|-----------|-------------|
| Retuning blend weights on the 58-study gold set | Public dropped to **0.954**; offline gains did not transfer |
| Stacking multiple additional Raptor-384 models with S6 | Public notebooks completed, but official submits timed out on the hidden set |
| Low-quality pseudo-label models | No reliable gain relative to the public ensemble |
| Meniscus/Fracture specialist fine-tunes | Weaker than the baseline on gold meniscus labels; not used |

### Stage 3 — Final selection

The public leaderboard contained a large cluster at **0.958**. On the private leaderboard, top scores were near **0.960**, while many 0.958 public entries landed around **0.942**. The selected submissions belong to that stable public cluster: they emphasize faithful reproduction of a strong ensemble and conservative additive components, rather than aggressive late-stage stacking.

---

## Takeaways

1. In code competitions with strong public ensembles, reproducing the competitive stack is often more important than early self-training.  
2. Weight search on a tiny gold subset can overfit and degrade the leaderboard score.  
3. Hidden-set runtime is part of the objective: a successful public dry-run does not guarantee a scored submission.  
4. Small-weight mixtures of closely related models are usually safer than large changes to the fusion formula.

---

## License

Documentation is released under MIT. Competition data and third-party model weights remain under their original terms and must not be redistributed here.

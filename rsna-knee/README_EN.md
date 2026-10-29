# RSNA Knee Abnormality Detection

> 中文版：[README.md](./README.md)

Kaggle · [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · Code Competition

- **Task**: Predict 12 clinically relevant knee findings from MRI studies (ACL / MCL / menisci / three-compartment OA / effusion / synovitis / Baker's cyst / contusion / fracture, etc.). Metric: **macro ROC-AUC** over the 12 labels.
- **Setting**: Code competition. Only ~1.3% of training studies have gold labels; the rest rely on report-derived weak labels and/or public checkpoints. Submissions are re-run on a hidden test set under a strict runtime limit.
- **Result**: **Private 592 / 6299 (Bronze)**; Private **0.942** / Public **0.958** (Public ≈ #575)
- **Team**: Solo

This folder is a post-competition writeup (approach, final blend, what worked / failed). It does **not** include competition data or third-party weight files. See [EXPERIMENTS.md](./EXPERIMENTS.md) for a compact experiment list.

---

## Final submissions

Both selected submissions sit on a community-reproducible **Public 0.958 strong baseline** (Private score 0.942 for both):

1. **S6 → Raptor-half** (primary)  
   R6 ConvNeXt (Helios `fe055406-w1`) ⊕ public Raptor CoAtNet (384 + 224, A224=0.35) ⊕ per-finding blend; then mix **S6** into the **Raptor side** with a fixed small weight `W_S6=0.12` (not into R6/compact). If S6 fails, fall back to the no-S6 baseline.
2. **Reader @0.10** (second selected)  
   Same 0.958 baseline, plus goodpjw’s 2.5D ConvNeXt reader at weight 0.10.

Baseline structure:

```text
R6 (ConvNeXt + flip TTA)
    ⊕
Raptor-384 SWA   (publication CoAtNet; OAI-blendjev lineage)
    ⊕
Raptor-224       (same family; A224≈0.35 into the Raptor side)
    ⊕
per-finding W_RAPTOR_C   (frozen; no further gold-58 retuning)
```

---

## Approach

### 1. Match the public strong baseline first

Early self-training / weak-label fusion stalled around 0.94x. Reaching **Public 0.958** came from faithfully reproducing the Helios / nartaa / goodpjw stack (true R6-w1 + dual-resolution Raptor + per-finding blend). After that, every change had a hard constraint: **do not go below Public 0.958**.

### 2. Extra models must not hurt—and must fit the hidden runtime

When adding models on top of 0.958, the safer pattern was:

- Add only **small-weight** near-cousin CoAtNets (e.g. coat-extra `v10geo` @0.15, G233@320), and avoid retuning `W_RAPTOR_C`
- Mix S6 into the **Raptor side** (W=0.12); mixing into compact/R6 (publicly ~0.957) was worse

What to avoid:

| Attempt | Outcome |
|---------|---------|
| Grid-search global / per-label weights on noisy gold-58 | Public dropped to **0.954** (offline gains did not transfer) |
| S6 + multiple Raptor-384 arms (triple / dual-extra / fullcohort, …) | Public notebook finished, but **official submit timed out** on the hidden set |
| Low-quality pseudo-label weights (e.g. Mycelis, gold≈0.60) | No reliable gain vs baseline → rejected |
| Meniscus/Fracture specialist (weaker than baseline on gold meniscus) | Deprioritized; not used in final seats |

### 3. Reading the private leaderboard

Many teams clustered at **Public 0.958**. Private leaders were around **0.960**, while the 0.958 public band mostly landed near **~0.942**. The bronze came from a stable reproduction of that strong baseline with careful additive arms—not from last-minute heavy stacking.

---

## Timeline (compressed)

| Stage | What | Public |
|-------|------|--------|
| Match public stack | R6-w1 + Raptor384/224 + per-finding | **0.958** |
| Failed reweight | gold-58 reweight-cons | 0.954 |
| Useful tweaks | S6→Raptor-half; coat-extra v10geo; reader-0.1 | 0.958 (tie / finer digits) |
| Timeout stacks | S6 + multi R384 / fullcohort, … | Submit failed (runtime) |
| Light probes | G233@320, etc. | 0.958 (no lift past 0.958) |

---

## Reproduction notes (no data)

To reproduce a same-structure submission in a Kaggle code environment, typical public assets include:

- `heliosli/rsna-knee-r6-frozen-fe055406-w1`
- `nartaa/rsna-knee-publication-swa-weights-20261007`
- `nartaa/rsna-knee-s6fall-weights` (S6)
- `goodpjw2008/rsna-knee-2-5d-convnext-reader` (reader, optional)
- Optional near-cousin weights: `manatoshoji03/rsna-knee-coat-extra(2)`, etc.

Blend outline:

1. Rank Raptor-384 and Raptor-224 separately, then combine with A224 into the Raptor side  
2. Rank-blend with R6 using per-finding `W_RAPTOR_C`  
3. (Primary) Mix S6 ranks into the Raptor side at `W_S6=0.12`; skip on failure  
4. Keep the number of inference arms small enough to pass **hidden runtime**

Local EffB0-MIL / consensus residual students were tried; they did not beat the baseline’s weak labels reliably and were **not** used in the final selected submissions.

---

## Takeaways

1. In code competitions, a **reproducible public strong stack** often beats early self-training; match the baseline before hunting deltas.  
2. Offline reweighting on a tiny gold set overfits easily and can hurt the leaderboard.  
3. Budget **hidden runtime** before adding arms: a green public run ≠ a scored submission.  
4. Small-weight near-cousin blends (same-family CoAtNet / S6→Raptor-half) are usually safer than aggressive formula changes.

---

## License

Writeup under MIT. Competition data and third-party weights remain under their original terms—do not redistribute.

# RSNA Knee Abnormality Detection

> English: [README_EN.md](./README_EN.md)

Kaggle · [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · Code Competition

| | |
|---|---|
| **任务** | 膝关节 MRI 多标签异常检测（12 个临床发现） |
| **指标** | Macro-averaged ROC-AUC |
| **成绩** | Private **592 / 6299**（Bronze）· Private **0.942** / Public **0.958** |
| **形式** | 个人参赛 |

本目录记录赛后整理的最终方案与主要实验结论。竞赛数据与第三方模型权重不在此仓库分发；更细的实验条目见 [EXPERIMENTS.md](./EXPERIMENTS.md)。

---

## 问题设定

任务是根据膝关节 MRI 检查预测十二项临床相关发现，包括韧带损伤（ACL / MCL）、半月板撕裂、三间室骨关节炎、积液、滑膜炎、Baker 囊肿、骨挫伤与骨折等。训练集中仅有少量研究带有官方金标，其余样本通常依赖报告派生的弱标签，或借助公开模型与权重。

本赛为 **Code Competition**：提交需在隐藏测试集上重新执行，并受运行时间限制。因此方案设计需要在预测效果与推理开销之间取得平衡。

---

## 最终方案

### 选定提交

最终评分选用同一公开集成的两个互补变体。二者均为 Public **0.958**、Private **0.942**。

1. **主提交 — S6 混入 Raptor 分支**  
   以公开的 R6 ⊕ 双分辨率 Raptor 集成及按发现融合权重为基础，将 S6 ConvNeXt 以固定权重 `W_S6 = 0.12` 混入 **Raptor 侧**。若 S6 推理失败，则回退为不含 S6 的集成。

2. **备选提交 — Reader @ 0.10**  
   保持同一 R6 ⊕ Raptor 集成，并将 goodpjw 的 2.5D ConvNeXt reader 以权重 `0.10` 混入。

### 集成主干

```text
R6 ConvNeXt  (Helios fe055406-w1, flip TTA)
        ⊕
Raptor-384 SWA  (nartaa publication CoAtNet)
        ⊕
Raptor-224      (同系；A224 ≈ 0.35)
        ⊕
按发现融合权重 W_RAPTOR_C
```

最终方案使用的公开权重包括：

- `heliosli/rsna-knee-r6-frozen-fe055406-w1`
- `nartaa/rsna-knee-publication-swa-weights-20261007`
- `nartaa/rsna-knee-s6fall-weights`
- `goodpjw2008/rsna-knee-2-5d-convnext-reader`（备选提交）

近亲 CoAtNet 权重（如 `manatoshoji03/rsna-knee-coat-extra`）曾作为小权重附加项评估，未能在 Public 0.958 之上取得进一步提升。

### 推理融合

1. 对 Raptor-384 与 Raptor-224 预测做秩归一化，再以系数 `A224` 融合。  
2. 将 Raptor 分支与 R6 按发现权重 `W_RAPTOR_C` 做秩融合。  
3. 主提交中，将 S6 的秩分数以 `W_S6 = 0.12` 混入 Raptor 分支。  
4. 控制前向次数，以满足隐藏集运行时间限制。

---

## 方法概要

### 阶段一：复现有竞争力的公开集成

早期自训练与弱标签管线达到约 0.94 中段，但尚未追上最强公开 Notebook。因此第一阶段优先完整复现公开栈：R6、双分辨率 Raptor CoAtNet，以及按发现融合。该复现稳定取得 Public **0.958**，并作为后续工作的参照基准。

### 阶段二：在约束下寻找可叠加增益

后续工作集中于小幅、可控的改动：在保持 Public 0.958 水平的同时，满足 Code Competition 的运行时间预算。主要尝试包括：

- 将 S6 混入 Raptor 分支，而非混入 R6/compact（后者在公开复现路径上约 0.957）。  
- 在不重调 `W_RAPTOR_C` 的前提下，小权重加入近亲 CoAtNet 检查点。  
- 测试额外 CoAtNet 分支时优先采用较轻分辨率（如 320）。

线下看起来有希望、但线上或提交阶段失败的方向予以舍弃，主要包括：

| 方向 | 观察 |
|------|------|
| 在 58 例金标集上重调融合权重 | Public 降至 **0.954**；线下收益未能迁移 |
| 叠加多个额外 Raptor-384 模型并加入 S6 | 公开 Notebook 可跑完，但正式提交在隐藏集超时 |
| 低质量伪标签模型 | 相对公开集成未见稳定收益 |
| Meniscus / Fracture 专项微调 | 金标半月板上弱于基线；未采用 |

### 阶段三：最终选型

Public 榜上大量提交聚集在 **0.958**。Private 前列约在 **0.960** 附近，而许多 Public 0.958 的方案最终落在约 **0.942**。本次选定提交属于该稳定公开簇：更强调对强集成的忠实复现与保守叠加，而非赛末激进堆栈。

---

## 结论

1. 在存在强公开集成的 Code Competition 中，完整复现往往比早期自训练更关键。  
2. 在极小金标子集上搜索融合权重容易过拟合，并可能损害榜上分数。  
3. 隐藏集运行时间是目标的一部分；公开 Notebook 跑通并不保证正式提交可计分。  
4. 对近亲模型做小权重混合，通常比大幅改动融合公式更稳妥。

---

## License

文档以 MIT 发布。竞赛数据与第三方模型权重仍受其原始条款约束，请勿在此仓库二次分发。

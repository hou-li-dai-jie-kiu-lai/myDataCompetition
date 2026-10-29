# RSNA Knee Abnormality Detection

Kaggle · [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · Code Competition

- **任务**：从膝关节 MRI study 预测 12 个临床异常标签（ACL / MCL / 半月板 / 三间室 OA / Effusion / Synovitis / Baker's / Contusion / Fracture 等），指标为 **12 类 macro ROC-AUC**
- **设定**：代码赛；仅约 1.3% 训练 study 有金标，其余依赖报告弱标签 / 公开权重；提交需在 hidden 测试集重跑且受 runtime 限制
- **成绩**：**Private 592 / 6299（铜牌）**；Private **0.942** / Public **0.958**（Public ≈ #575）
- **形式**：Solo

本目录为赛后方案整理（思路、最终栈、有效/无效实验），不含赛题原始数据与第三方权重文件。

---

## 最终采用的栈

计分席最终落在「公开 0.958 地板」上的两个变体（私榜同为 0.942）：

1. **S6 → Raptor-half**（主选）  
   R6 ConvNeXt（Helios `fe055406-w1`）⊕ 公开 Raptor CoAtNet（384 + 224，A224=0.35）⊕ 按病种权重融合；额外把 **S6** 以固定小权重 `W_S6=0.12` 溶进 **Raptor 半边**（不是溶进 R6/compact）。S6 软失败则回退地板。
2. **Reader-0.1**（第二席）  
   同一 0.958 地板上，再叠 goodpjw 2.5D ConvNeXt reader @0.10。

核心地板本身来自社区可复现配方：

```text
R6 (ConvNeXt, flip TTA)
    ⊕
Raptor-384 SWA  (publication CoAtNet, OAI-blendjev 血统)
    ⊕
Raptor-224      (同族，A224≈0.35 进 Raptor 半边)
    ⊕
per-finding W_RAPTOR_C  (固定，未再 gold 拧)
```

---

## 方法概览

### 1. 先对齐公开地板，再谈增益

早期自训 / 弱标签融合能到 0.94x，但真正拉开到 **Public 0.958** 的是对齐 Helios / nartaa / goodpjw 一线公开栈（真 R6-w1 + 双分辨率 Raptor + 按病融合）。之后所有尝试都以「不掉下 0.958 地板」为硬约束。

### 2. 异构臂要「非负」且过 hidden runtime

在 0.958 上继续加模型时，有效策略是：

- **只加小权重异构 CoAtNet**（如 coat-extra `v10geo` @0.15、G233@320），且尽量 **不改** `W_RAPTOR_C`
- **S6 溶进 Raptor-half**（W=0.12）比溶进 compact/R6（公开复现约 0.957）更稳

需要避免的：

| 做法 | 结果 |
|------|------|
| 用 noisy gold-58 网格拧全局 / 按病权重 | Public 掉到 **0.954**（离线 Δ 不兑现） |
| S6 + 多路 R384（triple / dual-extra / fullcohort 等） | 公开 dry-run 能完，**提交 hidden 超时** |
| 低质量伪标族（如 Mycelis gold≈0.60） | 直接拒绝叠入 |
| Meniscus/Fracture specialist（gold 半月板弱于地板） | 降优先，未进最终席 |

### 3. 最终私榜解读

公开榜大量队伍挤在 **0.958**；私榜头部约 **0.960**，0.958 平台普遍落到 **~0.942**。铜牌落在该平台的稳定复现上，而不是靠最后几天的重栈赌分。

---

## 关键时间线（压缩）

| 阶段 | 内容 | 公开分 |
|------|------|--------|
| 对齐公开栈 | R6-w1 + Raptor384/224 + per-finding | **0.958** |
| 无效拧权 | gold-58 reweight-cons | 0.954 |
| 有效细调 | S6→Raptor-half；coat-extra v10geo；reader-0.1 | 0.958（细位/持平） |
| 超时重栈 | S6+多 R384 / fullcohort 等 | 提交失败（runtime） |
| 轻臂扫射 | G233@320 等 | 0.958（未超过地板） |

---

## 可复现要点（无数据版）

若要在 Kaggle 代码环境复现同结构提交，需要挂载的公开资产大致包括：

- `heliosli/rsna-knee-r6-frozen-fe055406-w1`
- `nartaa/rsna-knee-publication-swa-weights-20261007`
- `nartaa/rsna-knee-s6fall-weights`（S6 臂）
- `goodpjw2008/rsna-knee-2-5d-convnext-reader`（reader 臂，可选）
- 可选异构：`manatoshoji03/rsna-knee-coat-extra(2)` 等

融合逻辑要点：

1. Raptor-384 / Raptor-224 各自 rank 后按 A224 合成 Raptor 半边  
2. 与 R6 按病种 `W_RAPTOR_C` 做 rank blend  
3. （主选）再把 S6 rank 以 `W_S6=0.12` 溶进 Raptor 半边；失败则跳过  
4. 提交前控制总推理臂数，优先保证 **hidden runtime** 过线

本地训练侧曾尝试 EffB0-MIL / 共识软标残差学生等，gold gate 未稳定超过地板弱项，**未进入最终计分席**。

---

## 经验总结

1. **代码赛里，公开可复现强栈 > 早期自训增益**；先复现地板再谈差分。  
2. **gold-58 离线拧权极易过拟合**，线上可直接倒退。  
3. **加臂前先量 hidden 时限**：dry-run COMPLETE ≠ 提交可评分。  
4. 小权重异构（同族 CoAtNet / S6→Raptor-half）比大改融合公式更安全。

---

## License

文档与整理脚本 MIT。赛题数据与第三方权重遵循原竞赛 / 数据集条款，请勿二次公开分发。

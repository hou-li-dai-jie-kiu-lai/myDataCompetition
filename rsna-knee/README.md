# RSNA Knee Abnormality Detection

> English version: [README_EN.md](./README_EN.md)

Kaggle · [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · 代码赛

- **任务**：从膝关节 MRI study 预测 12 个临床异常标签（ACL / MCL / 半月板 / 三间室 OA / Effusion / Synovitis / Baker's / Contusion / Fracture 等），指标为 **12 类 macro ROC-AUC**
- **设定**：代码赛；仅约 1.3% 训练 study 有金标，其余依赖报告弱标签 / 公开权重；提交需在隐藏测试集上重跑，并受 runtime 限制
- **成绩**：**Private 592 / 6299（铜牌）**；Private **0.942** / Public **0.958**（Public ≈ #575）
- **形式**：Solo

本目录为赛后方案整理（思路、最终融合、有效/无效实验），不含赛题原始数据与第三方权重文件。更细的实验列表见 [EXPERIMENTS.md](./EXPERIMENTS.md)。

---

## 最终提交

计分使用的两份提交，都建立在社区可复现的 **Public 0.958 强基线**之上（Private 同为 0.942）：

1. **S6 → Raptor-half**（主选）  
   R6 ConvNeXt（Helios `fe055406-w1`）⊕ 公开 Raptor CoAtNet（384 + 224，A224=0.35）⊕ 按病种权重融合；再将 **S6** 以固定小权重 `W_S6=0.12` **融入 Raptor 侧**（而不是融入 R6 / compact）。若 S6 推理失败，则回退到不含 S6 的基线。
2. **Reader @0.10**（第二席）  
   同一 0.958 基线上，再混入 goodpjw 2.5D ConvNeXt reader，权重 0.10。

基线结构：

```text
R6 (ConvNeXt + flip TTA)
    ⊕
Raptor-384 SWA   (publication CoAtNet；训练血统含 OAI-blendjev)
    ⊕
Raptor-224       (同族；A224≈0.35 与 384 合成为 Raptor 侧)
    ⊕
per-finding W_RAPTOR_C   (固定；不再用 gold-58 重调)
```

---

## 方法概览

### 1. 先对齐公开强基线，再谈增益

早期自训 / 弱标签融合大约停在 0.94x。真正到 **Public 0.958**，靠的是对齐 Helios / nartaa / goodpjw 一线公开方案（真 R6-w1 + 双分辨率 Raptor + 按病融合）。之后所有改动都以「**公开分不低于 0.958**」为硬约束。

### 2. 额外模型要「不拖后腿」，且过得了隐藏集时限

在 0.958 上继续加模型时，较稳的做法是：

- 只加 **小权重** 的同族 / 近亲 CoAtNet（如 coat-extra `v10geo` @0.15、G233@320），并尽量 **不改** `W_RAPTOR_C`
- 把 S6 **融入 Raptor 侧**（W=0.12）；若改成融入 compact/R6（公开复现约 0.957），往往更差

需要避免的：

| 做法 | 结果 |
|------|------|
| 用噪声较大的 gold-58 网格重调全局 / 按病权重 | Public 掉到 **0.954**（离线涨分无法兑现） |
| S6 + 多路 Raptor-384（triple / dual-extra / fullcohort 等） | 公开 notebook 能跑完，**正式提交在隐藏集超时** |
| 低质量伪标权重（如 Mycelis，gold≈0.60） | 相对基线无可靠增益 → 不加入融合 |
| Meniscus/Fracture specialist（gold 上半月板弱于基线） | 优先级下调，未进最终席 |

### 3. 私榜怎么读

公开榜有大量队伍挤在 **0.958**；私榜头部大约 **0.960**，而 0.958 这一档普遍落到 **~0.942**。铜牌主要来自对该强基线的稳定复现与谨慎加臂，而不是赛末盲目堆模型。

---

## 关键时间线（压缩）

| 阶段 | 内容 | Public |
|------|------|--------|
| 对齐公开方案 | R6-w1 + Raptor384/224 + per-finding | **0.958** |
| 无效重加权 | gold-58 reweight-cons | 0.954 |
| 有效微调 | S6→Raptor-half；coat-extra v10geo；reader-0.1 | 0.958（持平或细分数略优） |
| 超时堆叠 | S6 + 多路 R384 / fullcohort 等 | 提交失败（runtime） |
| 轻量试探 | G233@320 等 | 0.958（未超过 0.958） |

---

## 复现要点（不含数据）

在 Kaggle 代码环境复现同结构提交，大致需要挂载：

- `heliosli/rsna-knee-r6-frozen-fe055406-w1`
- `nartaa/rsna-knee-publication-swa-weights-20261007`
- `nartaa/rsna-knee-s6fall-weights`（S6）
- `goodpjw2008/rsna-knee-2-5d-convnext-reader`（reader，可选）
- 可选近亲权重：`manatoshoji03/rsna-knee-coat-extra(2)` 等

融合顺序要点：

1. Raptor-384 / Raptor-224 各自做 rank，再按 A224 合成 Raptor 侧  
2. 与 R6 按病种 `W_RAPTOR_C` 做 rank blend  
3. （主选）再将 S6 rank 以 `W_S6=0.12` 融入 Raptor 侧；失败则跳过  
4. 控制推理模型数量，优先保证 **隐藏集 runtime** 通过

本地还尝试过 EffB0-MIL、共识软标残差学生等；相对基线弱项未形成稳定正增益，**未进入最终计分席**。

---

## 经验总结

1. **代码赛里，可复现的公开强方案往往比早期自训更关键**；先对齐基线，再做差分。  
2. **小规模金标上的离线重加权很容易过拟合**，线上可能直接掉分。  
3. **加模型前先估隐藏集时限**：notebook 跑通 ≠ 提交能出分。  
4. 小权重近亲融合（同族 CoAtNet / S6→Raptor-half）通常比大幅改融合公式更安全。

---

## License

文档 MIT。赛题数据与第三方权重遵循原竞赛 / 数据集条款，请勿二次公开分发。

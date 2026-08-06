# P88 方法说明

## 一句话

在多条「合法单井 DNA」的 OOF 预测上，做 **对 P20H 残差的多 α Ridge 袋装**，再与 **P69** 做折内权重融合，最后加 **warm-up + (TVT+Z) 投影** 后处理。

本地投影 OOF ≈ **8.40**；公榜 **7.841**；私榜 **8.836**（我们最终计分对应的最好私榜提交）。

## 流水线层级（概念）

```text
轨迹 + GR + typewell
        │
        ▼
硬尺子 / PF 集成 (raw·aff·dual, Ridge-hard 等)     → hard, stock, …
        │
        ▼
CatBoost 残差族
  · P20H  : Ridge-hard + Cat          （公榜主 hold）
  · P12F  : modescan / 双模态软特征
  · P50e  : shape-hard 线
  · P69   : 在 P50e 上再 Stage2 Cat
        │
        ▼
P88 core5
  X = [p20h, p12f, p50e, hard, stock, md_since, p69]
  对每个 GroupKFold：
    多 α∈{0.3,1,3,10,30} Ridge 拟合 (y - p20h)
    预测 = p20h + mean_α Ridge(X)
  再折内网格搜索 w，使 w*p69 + (1-w)*p81b 在折训练井上最优
        │
        ▼
PP: delta warm-up (τ≈85) + 井级 U=TVT+Z 投影 (blend≈0.7)
        │
        ▼
submission.csv  (绝对 TVT = last_known + delta)
```

## 为何叫 core5

候选残差通道固定为五条井内预测：`p20h, p12f, p50e, hard, stock`，再拼 `md` 与 `p69`。  
**不是**把已提交成品做黑盒加权撞车；Ridge 学的是相对 P20H 的残差结构。

## 复现本仓库能提供的部分

完整 Cat/PF 训练依赖大规模特征缓存与权重，**不在本仓库发布**。  
这里提供：

- `rogii_share.p88_ridge_blend`：在你已有的 OOF 列上复现 **Ridge 袋装 + 折内 blend**
- `rogii_share.postprocess`：warm-up / 投影（需井 CSV 几何）
- `rogii_share.metrics` / `cv`：按井 RMSE 与 GroupKFold

你需要自备：按行对齐的 `y, well, md` 以及各通道 OOF 向量。

## 后处理（PP）

1. **Warm-up**：`delta *= 1 - exp(-md_since / τ)`，靠近已知前缀处少动。  
2. **投影**：在评价段对 `U = TVT + Z` 相对锚点做稳健多项式拟合，再与原预测 blend。

只使用 MD/Z/可见前缀几何 + 当前预测，不读评价段真值。

## 与 P20H / P104 的关系

| 线 | 角色 |
|----|------|
| P20H | 公榜最稳单 DNA；私榜一般 |
| P69 | P88 的融合锚之一 |
| P88 | 私榜最好（我们） |
| P104 | 离线更好，prefix-gate 公榜崩盘 → **不要迷信离线门控** |

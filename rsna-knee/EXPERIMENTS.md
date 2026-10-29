# RSNA Knee — 实验备忘（精简）

## 有效 / 持平

- **R6-w1 + Raptor384/224 + per-finding**：公开强基线 **0.958**
- **S6→Raptor-half (W=0.12)**：公开 0.958；细分数略优，进最终 #1 席
- **Reader @0.10**：公开 0.958，进最终 #2 席
- **coat-extra v10geo @0.15**（不改 `W_RAPTOR_C`）：公开 0.958
- **G233 coat320 @0.15 + S6**：公开 0.958（更轻，能过隐藏集时限）

## 无效 / 有害

- **gold-58 reweight-cons**：离线约 +0.009 → 线上 **0.954**
- Contusion / Fracture 等单病 `W_RAPTOR` 盲调：多为 **0.957**
- **S6→compact@0.3**（公开复现路径）：约 **0.957**
- Mycelis 等弱 OOF 族：相对基线无明显正增益 → 不加入融合
- Meniscus/Fracture specialist：gold 上半月板弱于基线 → 不优先

## 提交失败（runtime）

S6 + 多路 Raptor-384（triple / bs14 / dual-extra / angelus-v4 / fullcohort 等）在公开 notebook 可跑完，但正式提交报错：

> submission notebook exceeded the allowed runtime

教训：代码赛加模型必须给隐藏集留足时间余量。

## 最终私榜

多数 Public 0.958 的提交，Private 均为 **0.942**；最终计分席未因赛末堆叠而改变。

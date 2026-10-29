# RSNA Knee — 实验记录

## 有效或与公开基线持平

- **R6-w1 + Raptor-384/224 + 按发现融合**：Public **0.958**（参照集成）
- **S6 → Raptor 半边（`W_S6=0.12`）**：Public 0.958；最终主提交
- **Reader @ 0.10**：Public 0.958；最终备选提交
- **coat-extra v10geo @ 0.15**（不改 `W_RAPTOR_C`）：Public 0.958
- **G233 CoAtNet-320 @ 0.15 + S6**：Public 0.958；更轻的附加分支，可通过隐藏集时限

## 无效或有害

- **gold-58 重调融合权重**：线下约 +0.009 → Public **0.954**
- 盲目单标签改动 `W_RAPTOR`（如 Contusion / Fracture）：多为 **0.957**
- **S6 → compact @ 0.3**（公开复现路径）：约 **0.957**
- Mycelis 等弱 OOF 族：相对参照集成无稳定收益
- Meniscus / Fracture 专项：金标半月板上弱于基线；未采用

## 提交失败（运行时间）

将 S6 与多个额外 Raptor-384 检查点叠加（triple / bs14 / dual-extra / angelus-v4 / fullcohort 等）时，公开 Notebook 往往可跑完，但正式评分报错：

> submission notebook exceeded the allowed runtime

## Private 榜

多数 Public 0.958 的提交对应 Private **0.942**。赛末加重堆栈未能改善最终选定席位。

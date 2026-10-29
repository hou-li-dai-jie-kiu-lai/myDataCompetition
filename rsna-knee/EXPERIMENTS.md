# RSNA Knee — 实验备忘（精简）

## 有效 / 持平

- **R6-w1 + Raptor384/224 + per-finding**：公开地板 **0.958**
- **S6→Raptor-half (W=0.12)**：公开 0.958，细位更优，进最终 #1 席
- **Reader @0.10**：公开 0.958，进最终 #2 席
- **coat-extra v10geo @0.15**（无改 W_RAPTOR_C）：公开 0.958
- **G233 coat320 @0.15 + S6**：公开 0.958（轻臂，过 hidden 时限）

## 无效 / 有害

- **gold-58 reweight-cons**：离线 +0.009 → 线上 **0.954**
- Contusion / Fracture 等单病 W_RAPTOR 盲拧：多为 **0.957**
- **S6→compact@0.3**（公开复现路径）：约 **0.957**
- Mycelis 等弱 OOF 族：相对地板无正残差 → 不叠
- Meniscus/Fracture specialist：gold 半月板弱于地板 → 不优先

## 提交失败（runtime）

S6 + 多路 R384（triple / bs14 / dual-extra / angelus-v4 / fullcohort 等）在公开 kernel COMPLETE，但 CreateCodeSubmission 报：

> submission notebook exceeded the allowed runtime

教训：代码赛加臂必须给 hidden 留余量。

## 最终私榜

多数 0.958 公开提交的 Private 均为 **0.942**；计分席未因最后几天重栈改变。

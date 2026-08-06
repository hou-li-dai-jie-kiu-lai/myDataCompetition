# 赛题介绍

## 背景

[ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)  
水平井钻进时，专家需要解释井眼路径附近的地质位置。赛题用机器学习预测 **TVT（True Vertical Thickness）**，辅助识别有利层位、减少无效进尺。

Slogan：*Drilling a horizontal well is like navigating underground without a map.*

## 任务形式

- **Code competition**：必须用 Kaggle Notebook 出 `submission.csv`（提交重跑时通常断网）。
- 训练集：多口井，每口井有水平井轨迹 CSV、typewell、可选剖面图。
- 测试集：隐藏约 200 口井；页面上的 `test/` 只是样例，**不能当真实 LB 代理**。

## 评分

提交列：`id,tvt`。指标为整表 **RMSE**（越低越好）：

\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i (y_i-\hat y_i)^2}
\]

## 关键约束（务必遵守）

1. **按井分组验证**：用 `GroupKFold` / 按 `well` 划分，禁止把同一口井的点拆进 train/val。
2. **禁止同井 train↔test 接触覆写**（同编号井直接拷贝解释）——私榜会失效，也违反规则精神。
3. 可见前缀 `TVT_input` 可用；评价段 `TVT` 在测试中为 NaN。

## 时间线（已结束）

- Start: 2026-05-05  
- Final submission: 2026-08-05  

奖金池见竞赛 Overview（前四名现金奖）。

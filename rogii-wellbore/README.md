# ROGII Wellbore Geology Prediction

Kaggle · [rogii-wellbore-geology-prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

- **任务**：预测水平井侧钻段 **TVT**（RMSE ↓）
- **数据**：每口井一对 CSV — `{well}__horizontal_well.csv` + `{well}__typewell.csv`
- **CV**：GroupKFold × 5（按井分组）
- **成绩**：Private **8.836** / Public **7.841**（约第 422）

本目录提供可运行的完整训练与推理代码（不含赛题数据包与已训练权重）。

---

## 方法概览

| 阶段 | 内容 |
|------|------|
| 特征 | 粒子滤波（PF）多尺度 TVT tracks：raw / affine / dual；附带几何与 GR 等列 |
| Hard | `Ridge(α=1, positive=True)` 拟合 tracks |
| Main Cat | 在 hard 上拟合残差（CatBoost） |
| Alt Cat | 第二通道（去掉 dual tracks），同残差设定 |
| Anchor | 第二阶段 Cat 拟合 `(y − main)`，训练集 shrink ≤ 1.6 |
| Stack | 多 α Ridge bag `{0.3,1,3,10,30}`，再与 anchor 折内加权融合 |
| 后处理 | MD warm-up（τ=85）+ `U = TVT + Z` 投影（blend=0.7） |

默认超参集中在 `src/rogii/config.py`。完整复现请用默认参数在全量 `train/` 上训练；`--fast` 仅用于冒烟，结果不可比。

> 竞赛过程中还实验过 modescan / beam / bi\_\* 等增广特征。本仓库以 PF tracks + 几何为主干，并给出与公开成绩对应的超参；跑通上述流水线即可得到同结构提交。

---

## 环境与数据

```bash
cd rogii-wellbore
pip install -r requirements.txt

kaggle competitions download -c rogii-wellbore-geology-prediction -p ./data
# 解压后目录应含 train/、test/、sample_submission.csv
export ROGII_DATA_ROOT=/path/to/data
```

Windows PowerShell 可用 `$env:ROGII_DATA_ROOT = "..."`。

---

## 运行

```bash
# 无竞赛数据：仅验证 stacking / 后处理逻辑
python scripts/02_demo.py

# 训练（默认 64 seeds × 250 particles，耗时较长）
python scripts/04_train.py --data $ROGII_DATA_ROOT
# 可选：CatBoost GPU
python scripts/04_train.py --data $ROGII_DATA_ROOT --gpu

# 推理 → outputs/submission.csv
python scripts/05_infer.py --data $ROGII_DATA_ROOT

# 冒烟（缩小 PF / Cat / 井数，勿用来对标成绩）
python scripts/04_train.py --data $ROGII_DATA_ROOT --fast
python scripts/05_infer.py --data $ROGII_DATA_ROOT --fast
```

可选：

- `scripts/01_eda.py`：轻量 EDA（需数据）
- `scripts/03_run_stack.py`：对已有 OOF 通道 NPZ 单独跑 stacking

产物写入 `outputs/`（已 gitignore）：`topline_artifact.joblib`、`submission.csv`。

---

## 代码结构

```text
src/rogii/
  config.py       默认超参
  features_pf.py  粒子滤波 tracks
  dataset.py      井 CSV → 特征表
  train.py        Hard + Cat + Anchor + Stack
  infer.py        折平均 + 后处理 + submission
  stack.py / postprocess.py / cv.py / metrics.py / ...
scripts/
  01_eda.py  02_demo.py  03_run_stack.py  04_train.py  05_infer.py
```

---

## 注意

- 必须按井分组交叉验证；禁止用同井标签泄漏到验证/测试
- 赛题数据受竞赛规则约束，请勿二次公开分发

## License

代码 MIT。

# ROGII Wellbore Geology Prediction

Kaggle · [rogii-wellbore-geology-prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

- **任务**：预测水平井侧钻段 **TVT**（RMSE ↓）
- **数据**：每口井一对 CSV — `{well}__horizontal_well.csv` + `{well}__typewell.csv`
- **CV**：GroupKFold × 5（按井分组）
- **成绩**：Private Ranking **422 / 6191（铜牌）**；Private **8.836** / Public **7.841**

本目录提供可运行的完整训练与推理代码（不含赛题数据包与已训练权重）。

---

## 整体方案结构

端到端顺序如下（训练与推理共用同一套特征与后处理）：

```text
井 CSV（horizontal + typewell）
        │
        ▼
① 特征：粒子滤波 PF tracks
   · raw / affine / dual（多尺度）
   · 几何 & GR 等辅助列
        │
        ▼
② Hard：Ridge(α=1, positive) 拟合 tracks → hard / stock
        │
        ├──────────────────┐
        ▼                  ▼
③ Main Cat              ④ Alt Cat
   hard 上残差              去掉 dual tracks 的残差通道
        │                  │
        └────────┬─────────┘
                 ▼
⑤ Anchor：Stage-2 Cat 拟合 (y − main)，再 shrink ≤ 1.6
                 │
                 ▼
⑥ Stack：多 α Ridge bag {0.3,1,3,10,30}
         ⊕ 与 Anchor 折内加权融合
                 │
                 ▼
⑦ 后处理：MD warm-up（τ=85）
         + U=TVT+Z 投影（blend=0.7）
                 │
                 ▼
绝对 TVT = last_known + delta  →  submission.csv
```

交叉验证一律 **按井分组**（GroupKFold），折内只在训练井上选 shrink / blend 权重，再应用到验证井。

| 步骤 | 模块 | 作用 |
|------|------|------|
| ① | `features_pf` / `dataset` | 从井 CSV 生成 PF tracks 与行级特征表 |
| ② | Hard Ridge | 给出可解释的硬基准 `hard` |
| ③④ | Main / Alt CatBoost | 两路残差树模型，增加多样性 |
| ⑤ | Anchor Cat | 校正 Main 的系统性残差 |
| ⑥ | Stack | 多正则 Ridge 袋装 + 与 Anchor 融合 |
| ⑦ | Post-process | 侧钻段起始段衰减 + 轨迹连续性投影 |

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
  features_pf.py  ① 粒子滤波 tracks
  dataset.py      ① 井 CSV → 特征表
  train.py        ②–⑥ Hard + Cat + Anchor + Stack
  infer.py        折平均 + ⑦ 后处理 + submission
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

# ROGII Wellbore Geology Prediction

Kaggle：[rogii-wellbore-geology-prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

一条 topline 的**完整可复现流水线**（特征 → 训练 OOF → stacking → 后处理 → 推理提交）。  
不含赛题数据包、不含训练好的权重文件、不含 API token。

| 项 | 值 |
|----|-----|
| 指标 | RMSE ↓ |
| 我们私榜参考 | 8.836（最终约第 422） |
| 验证协议 | 按井 `GroupKFold` |

---

## 赛题与数据

预测水平井侧钻段 **TVT**。每口井：`{well}__horizontal_well.csv` + `{well}__typewell.csv`。  
评价段 `TVT_input` 为 NaN；训练井上仍有 `TVT` 标签。

```bash
kaggle competitions download -c rogii-wellbore-geology-prediction -p ./data
# 解压后设置：
export ROGII_DATA_ROOT=/path/to/data   # 内含 train/ test/ sample_submission.csv
```

---

## Topline（唯一主线）

1. **PF 特征**：raw / heel-affine / dual 多 seed 粒子滤波 → 多尺度 track（残差相对 last_known）  
2. **Hard**：track 上 `Ridge(positive=True)`  
3. **Main / Alt**：CatBoost 残差回归（Alt 去掉 dual tracks）  
4. **Anchor**：Stage2 CatBoost 拟合 `(y - main)`  
5. **Stack**：多 α Ridge 袋装残差 + 与 anchor 折内融合  
6. **PP**：warm-up（τ=85）+ `U=TVT+Z` 投影  
7. **提交**：`tvt = last_known + delta`

> 说明：竞赛中还会用更重的 DTW/beam/modescan 等；本仓库给出的是**同一拓扑、可独立跑通**的完整实现。加大 `--n-seeds` / 井数可逼近更强特征设定。

---

## 代码结构

```text
src/rogii/
  features_pf.py   粒子滤波与 track
  dataset.py       从 CSV 构建训练/推理表
  train.py         Hard + Cat + Anchor + Stack
  infer.py         折平均推理 + PP + submission
  metrics.py / cv.py / postprocess.py / stack.py / pipeline.py
scripts/
  01_eda.py        数据概览
  02_demo.py       无数据合成冒烟（只测 stack）
  04_train.py      完整训练
  05_infer.py      完整推理
```

---

## 运行

```bash
cd rogii-wellbore
pip install -r requirements.txt

# 无数据冒烟
python scripts/02_demo.py

# 完整训练（示例：小规模快速验证）
python scripts/04_train.py --data $ROGII_DATA_ROOT --max-wells 20 --n-seeds 4 --n-particles 80 --n-splits 3 --cat-iterations 150

# 推理（使用 outputs/topline_artifact.joblib）
python scripts/05_infer.py --data $ROGII_DATA_ROOT --n-seeds 4 --n-particles 80
```

全量训练去掉 `--max-wells`，并提高 `--n-seeds`（如 16–64）、`--n-particles`（如 200–250）。

产物默认写在 `outputs/`（已 gitignore）：`topline_artifact.joblib`、`submission.csv`。

---

## 注意

- 必须按井分组；禁止同井 train 解释直接覆写 test  
- 前缀启发式 gate 容易离线好看、线上翻车；本 topline **不依赖**这类 gate  
- 数据受竞赛规则约束，勿公开二次分发

## License

代码 MIT。

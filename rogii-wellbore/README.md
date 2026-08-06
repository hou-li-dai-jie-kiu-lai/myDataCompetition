# ROGII Wellbore Geology Prediction

Kaggle：[rogii-wellbore-geology-prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

一条 topline 的完整可复现流水线。默认超参对齐我们**最终私榜提交**设定。

| 项 | 值 |
|----|-----|
| 私榜 | **8.836**（最终约第 422） |
| 公榜（同提交） | 7.841 |
| 本地 OOF proj（原实验记录） | ≈ 8.40 |
| 指标 | RMSE ↓ |
| CV | GroupKFold × 5（按井） |

不含赛题数据包、不含已训练权重、不含 API token。

---

## 赛题与数据

预测水平井侧钻段 **TVT**。每口井：`{well}__horizontal_well.csv` + `{well}__typewell.csv`。

```bash
kaggle competitions download -c rogii-wellbore-geology-prediction -p ./data
export ROGII_DATA_ROOT=/path/to/data   # 含 train/ test/ sample_submission.csv
```

---

## Topline（与最终结果对齐）

默认写在 `src/rogii/config.py`：

| 模块 | 最终设定 |
|------|----------|
| PF | **64 seeds × 250 particles**；scales `(3,5,8,12)`；dual `±15`；raw / affine / dual tracks |
| Hard | `Ridge(alpha=1, positive=True)` on tracks |
| Main Cat | 对 hard 的残差；`8000` iter, depth `7`, lr `0.025`, `od_wait=300` |
| Alt Cat | 第二通道（去 dual tracks）；同量级、seed=7 |
| Anchor | Stage2 Cat 拟合 `(y-main)` + train 上 shrink≤**1.6** |
| Stack | α∈`{0.3,1,3,10,30}` Ridge bag ⊕ 与 anchor 折内融合 |
| PP | warm-up **τ=85** + U=TVT+Z 投影 **blend=0.7** |

> 仓库实现了同一拓扑与最终超参。竞赛全量特征里还有 modescan / beam / bi_* 等增广；本复现以 PF tracks + 几何为主干。要贴近最终分数请用**默认全量参数**在完整 `train/` 上跑，不要用 `--fast`。

---

## 运行

```bash
cd rogii-wellbore
pip install -r requirements.txt

# 无数据冒烟（只测 stack）
python scripts/02_demo.py

# 最终设定训练（耗时长：全井 × 64×250 PF）
python scripts/04_train.py --data $ROGII_DATA_ROOT
# 可选 GPU：
python scripts/04_train.py --data $ROGII_DATA_ROOT --gpu

# 最终设定推理
python scripts/05_infer.py --data $ROGII_DATA_ROOT

# 仅管道冒烟（参数故意缩小，分数不可比）
python scripts/04_train.py --data $ROGII_DATA_ROOT --fast
python scripts/05_infer.py --data $ROGII_DATA_ROOT --fast
```

产物在 `outputs/`（gitignore）：`topline_artifact.joblib`、`submission.csv`。

---

## 代码结构

```text
src/rogii/
  config.py        最终超参（唯一真相来源）
  features_pf.py   PF tracks
  dataset.py       CSV → 特征表
  train.py         Hard + Cat + Anchor + Stack
  infer.py         折平均 + PP + submission
  metrics/cv/postprocess/stack/pipeline/...
scripts/
  01_eda.py  02_demo.py  04_train.py  05_infer.py
```

---

## 注意

- 必须按井分组；禁止同井 train 解释覆写 test  
- `--fast` 只用于验证代码可跑，**不能**对标私榜 8.836  
- 数据受竞赛规则约束，勿公开二次分发

## License

代码 MIT。

# ROGII Wellbore Geology Prediction

Kaggle 竞赛分享：[rogii-wellbore-geology-prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

本目录只保留 **一条 topline 方案** 的说明与可运行代码（评测 / 按井 CV / 后处理 / 残差 stacking）。  
**不含**赛题原始数据、模型权重、API token。

| 项 | 结果 |
|----|------|
| 指标 | RMSE（越低越好） |
| 私榜 | **8.836**（最终约第 422 名） |
| 验证 | 必须按井 `GroupKFold`，禁止同井接触拷贝 |

---

## 赛题在做什么

水平井钻进时，需要估计井眼路径附近的地质位置 **TVT**。  
每口井有：水平井轨迹 CSV（MD/X/Y/Z/GR、可见前缀 `TVT_input`）、typewell 垂向参考井。  
测试集评价段 `TVT` 为 NaN；页面上的 `test/` 只是样例，真实评分在隐藏集上重跑 Notebook。

提交格式：

```text
id,tvt
000d7d20_1442,11754.12
...
```

`id = {WELLNAME}_{row_index}`。

---

## 数据怎么拿

1. 在竞赛页同意规则  
2. 本机配置 Kaggle API（凭证放本机，不要写进仓库）  
3. 下载：

```bash
kaggle competitions download -c rogii-wellbore-geology-prediction -p ./data
# 解压后应有 data/train  data/test  data/sample_submission.csv
export ROGII_DATA_ROOT=/absolute/path/to/data
```

数据受竞赛规则约束，请勿公开二次分发。

---

## Topline 方法（唯一主线）

目标变量用 **相对最后已知 TVT 的残差**：

\[
y = \mathrm{TVT} - \mathrm{TVT}^{\mathrm{last\_known}}
\]

### 1) 底座预测（多通道）

在按井 GroupKFold 下，各自训出若干通道的 OOF 残差预测，例如：

- 硬尺子 / PF 集成通道（`hard`, `stock`）
- 主 CatBoost 通道（强单模）
- 形态 / modescan 等变体通道
- 再经一层浅 Stage2 得到的锚通道（下文称 `anchor`）

> 完整 PF + CatBoost 特征工程体积很大，本仓库不附权重；你需要自备各通道 OOF 向量。  
> stacking / 评测 / 后处理代码在本仓库是完整可跑的。

### 2) 残差 Ridge 袋装 + 与锚融合

对特征矩阵（示例列）：

`[main, alt_a, alt_b, hard, stock, md_since, anchor]`

在每个 GroupKFold：

1. 多 α Ridge 拟合 `(y - main)`，预测 `main + mean_α Ridge(X)`  
2. 在折的训练井上网格搜索权重 `w`，得到  
   `final = w * anchor + (1-w) * ridge_bag`

### 3) 后处理 PP

1. **Warm-up**：靠近可见前缀处缩小残差  
   `delta *= 1 - exp(-md_since / τ)`，默认 `τ=85`  
2. **U=TVT+Z 投影**：用井几何对绝对 TVT 做稳健多项式投影并 blend（默认 0.7）

### 4) 写出提交

`tvt = last_known + delta_pp`

---

## 代码结构

```text
src/rogii/
  dataio.py        读井 CSV、构造残差目标与基本字段
  metrics.py       全局 / 分井 RMSE
  cv.py            GroupKFold
  postprocess.py   warm-up + 投影
  stack.py         Ridge 袋装 + 锚融合（topline 核心）
  pipeline.py      把上述步骤串成一条清晰流水线
scripts/
  01_eda.py        有数据时做轻量 EDA
  02_demo.py       无数据合成冒烟（验证代码可跑）
  03_run_stack.py  对你的 OOF NPZ 跑完整 stacking + PP 摘要
```

---

## 快速开始

```bash
cd rogii-wellbore
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 无赛题数据：跑通 stacking / 评测逻辑
python scripts/02_demo.py

# 有数据：EDA
set ROGII_DATA_ROOT=D:\path\to\data
python scripts/01_eda.py

# 有自备 OOF 通道时（见 scripts/03_run_stack.py 注释）
python scripts/03_run_stack.py --npz path/to/channels.npz
```

### OOF NPZ 约定

`03_run_stack.py` 需要对齐长度的数组：

| key | 含义 |
|-----|------|
| `y` | 残差真值（仅训练评测用） |
| `wells` | 井 ID |
| `md` | `md_since` |
| `main` | 主通道 OOF |
| `alt_a`, `alt_b` | 辅助通道 OOF |
| `hard`, `stock` | 硬尺子通道 OOF |
| `anchor` | 锚通道 OOF |

---

## 关键注意

1. **按井分组**，不要点级随机切分  
2. **不要**用同编号井 train 解释直接覆写 test  
3. 可见前缀启发式（把侧钻预测「掰顺」回前缀）离线好看时，公榜/私榜可能反向——topline 不依赖这类 gate  
4. 公榜与私榜可能不一致；最终以私榜为准

## License

代码 MIT。赛题数据遵循 Kaggle 竞赛规则。

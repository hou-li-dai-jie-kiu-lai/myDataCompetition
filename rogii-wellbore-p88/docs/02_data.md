# 数据获取与结构

## 如何获取

1. 登录 Kaggle，打开竞赛页并 **Agree to rules**。
2. 安装 [Kaggle API](https://github.com/Kaggle/kaggle-api)，配置本机凭证（`~/.kaggle/kaggle.json` 或 `KAGGLE_API_TOKEN`）。**不要把 token 写进本仓库。**
3. 下载示例：

```bash
# 需已接受竞赛规则
kaggle competitions download -c rogii-wellbore-geology-prediction -p ./data
# 解压到 ./data ，使目录中出现 train/ test/ sample_submission.csv
```

或在 Notebook 中直接挂载竞赛数据路径 `/kaggle/input/...`。

建议环境变量：

```bash
export ROGII_DATA_ROOT=/absolute/path/to/competition/root
```

本仓库的 `scripts/run_eda.py` 会读该变量。

## 目录结构（概念）

```text
$ROGII_DATA_ROOT/
  sample_submission.csv
  train/
    {WELL}__horizontal_well.csv
    {WELL}__typewell.csv
    {WELL}.png                  # 可选
  test/                         # 公开仅为样例井
    {WELL}__horizontal_well.csv
    {WELL}__typewell.csv
```

## 水平井主表关键字段

| 字段 | 说明 |
|------|------|
| `MD` | Measured Depth（沿井筒） |
| `X,Y,Z` | 空间坐标；`Z` 为垂深 |
| `GR` | Gamma Ray（可有缺失） |
| `TVT` | 目标（训练有，测试评价段为 NaN） |
| `TVT_input` | 可见前缀 / 特征用 TVT；评价段为 NaN |
| 层界面列 | 多仅训练侧提供，推理勿当必有特征 |

`id` 提交格式：`{WELLNAME}_{row_index}`。

## Typewell

垂向参考井：`TVT` 作垂向索引 + `GR` + `Geology` 标签，用于剖面对比 / PF / DTW 等地质对齐特征（完整实现较长，本分享仓库给的是评测与 stacking 骨架，不捆绑巨型 PF 特征工程）。

## 许可提醒

数据受 **竞赛规则** 约束，仅可用于本竞赛相关用途；请勿把原始井 CSV 推到公开 GitHub。

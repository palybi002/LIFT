# 实验文件说明

本项目新增了“全量对比 + 消融 + 自动绘图”的标准流程，以下是每个文件的用途。

## 1. 思路文档
- `EXPERIMENT_PLAN.md`
  - 说明实验目标、对比矩阵、消融变量、执行顺序与产物路径。

## 2. 运行脚本
- `run_all_comparison_experiments.sh`
  - 一键跑对比实验。
  - 运行内容：
    - AirQuality/Weather：DLinear、DLinear+LIFT、LACFNet、PatchTST、Autoformer
    - ECL/Exchange/Sales：DLinear、DLinear+LIFT、LACFNet
  - 支持环境变量：
    - `SEQ_LEN`、`PRED_LEN`、`TRAIN_EPOCHS`、`ITR`、`GPU`、`ONLY_TEST`

- `run_ablation_experiments.sh`
  - 一键跑消融实验（默认在 AirQuality/Weather）。
  - 包含：
    - LACFNet: `top_k` = 1/3/5
    - LIFT: `leader_num` = 2/4/8（固定 `state_num=8`）
    - LIFT: `state_num` = 4/8/16（固定 `leader_num=4`）

## 3. 结果解析与图表
- `analyze_results.py`
  - 自动解析 `logs/*.log`。
  - 输出：
    - `comparison_results.csv`
    - `ablation_results.csv`
  - 自动提取字段：`Dataset`、`Model`、`MSE`、`MAE`、`Params`、`TrainTime`、消融参数等。

- `plot_experiment_results.py`
  - 读取 CSV 并生成图表到 `plots/`。
  - 图表类型：
    - 对比实验：各数据集 MSE/MAE/训练时间柱状图
    - 消融实验：各消融变量的 MSE/MAE 柱状图

## 4. 一键执行顺序
1. `bash run_all_comparison_experiments.sh`
2. `bash run_ablation_experiments.sh`
3. `python3 analyze_results.py`
4. `python3 plot_experiment_results.py`

## 5. 常见说明
- 为节省时间，脚本默认 `ONLY_TEST=true`，若已有 checkpoint 会直接测试；没有则按 `TRAIN_EPOCHS` 训练。
- 如果出现 `nan` 指标，多数是数据尺度或学习率导致，可在脚本里针对数据集降低 `learning_rate` 或增加预处理。

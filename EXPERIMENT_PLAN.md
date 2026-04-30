# 实验执行思路（全量对比 + 消融）

## 1. 目标
- 跑完所有对比实验：在目标数据集上比较多个模型的预测表现（MSE/MAE）与训练时间。
- 跑完消融实验：验证关键设计（如 LACFNet 的 top_k、LIFT 的 leader_num/state_num）对效果的影响。
- 自动整理结果：从日志提取指标，输出结构化 CSV，并生成图表到 `plots/`。

## 2. 实验范围
- 数据集：`AirQuality`、`Weather`、`ECL`、`Exchange`、`Sales`
- 对比模型：`DLinear`、`DLinear+LIFT`、`LACFNet`、`PatchTST`、`Autoformer`
- 统一设置：`seq_len=336`、`pred_len=96`、`itr=1`

## 3. 消融设计
- LACFNet 消融：`top_k` 分别为 `1/3/5`
- LIFT 消融：
  - `leader_num` 分别为 `2/4/8`
  - `state_num` 分别为 `4/8/16`
- 说明：消融以 `DLinear` backbone + `--lift` 进行，保证对比变量单一。

## 4. 执行顺序
1. 先跑对比实验脚本（包含 LIFT 所需 prefetch）
2. 再跑消融脚本（先 prefetch，再分别执行各组 ablation）
3. 统一解析日志生成 `comparison_results.csv` 和 `ablation_results.csv`
4. 生成图表并输出到 `plots/`

## 5. 产物约定
- 日志：`logs/*.log`
- 结构化结果：
  - `comparison_results.csv`
  - `ablation_results.csv`
- 图表：`plots/*.png`
- 说明文档：
  - 本文件：总体执行策略
  - `EXPERIMENT_README.md`：脚本与输出文件用途说明

## 6. 一键命令（最终）
- 全量对比：`bash run_all_comparison_experiments.sh`
- 消融实验：`bash run_ablation_experiments.sh`
- 解析和绘图：`python3 analyze_results.py && python3 plot_experiment_results.py`

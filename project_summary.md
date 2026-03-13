# LIFT 项目总结

## 项目简介
LIFT (Learning from Leading Indicators) 是一个用于多变量时间序列预测的 PyTorch 实现项目。该项目提出了一种新的视角，即通过利用变量之间的领先-滞后（lead-lag）关系来重新思考多变量时间序列中的通道依赖性（Channel Dependence）。

主要特点包括：
- **重新思考通道依赖性**：从领先-滞后关系的角度分析通道依赖性。
- **动态选择和移动指标**：为了缓解分布偏移，动态地选择领先指标并对其进行移动以与目标变量对齐。
- **LightMTS**：提出了一种参数高效的基准模型 LightMTS，在保持低参数量的同时实现了优异的性能。

## 关键 Python 文件作用

| 文件路径 | 作用描述 |
| :--- | :--- |
| `run_longExp.py` | 项目的主要入口脚本，用于运行长期预测实验。负责解析参数、加载数据、初始化模型并运行训练/测试循环。 |
| `run_prefetch.py` | 用于预计算领先指标（leading indicators）和领先步数（leading steps）的脚本。预计算可以加速训练过程。 |
| `models/LIFT.py` | 实现了 LIFT 模块。该模块作为一个包装器（Wrapper），可以接受一个骨干网络（Backbone）的预测结果，并利用 `LeadRefiner` 对其进行细化。 |
| `models/LightMTS.py` | 实现了 LightMTS 模型。这是一个轻量级的基准模型，使用简单的线性层作为骨干网络，并结合 LIFT 机制来捕捉通道依赖性。 |
| `exp/exp_main.py` | 包含了主要的实验逻辑，包括模型的训练、验证和测试过程。 |
| `exp/exp_lead.py` | 可能包含与领先指标相关的特定实验逻辑或辅助功能。 |
| `data_provider/data_loader.py` | 定义了数据加载器，负责读取和预处理时间序列数据集。 |
| `util/lead_estimate.py` | (推测) 包含用于估计领先-滞后关系和计算移动量的实用函数。 |

## 目录结构说明

```
.
├── data_provider/      # 数据加载和处理模块
│   ├── data_factory.py # 数据工厂模式实现
│   └── data_loader.py  # 自定义 Dataset 类
├── dataset/            # 存放数据集的目录
├── exp/                # 实验逻辑目录
│   ├── exp_basic.py    # 基础实验类
│   ├── exp_lead.py     # 领先指标相关实验逻辑
│   └── exp_main.py     # 主实验逻辑（训练、测试）
├── layers/             # 自定义神经网络层
├── models/             # 模型定义目录
│   ├── LIFT.py         # LIFT 核心模块
│   ├── LightMTS.py     # LightMTS 模型
│   ├── DLinear.py      # DLinear 模型
│   └── ...             # 其他对比模型（Autoformer, Informer 等）
├── scripts/            # 运行实验的 Shell 脚本
├── util/               # 工具函数目录
│   ├── lead_estimate.py # 领先指标估计工具
│   └── ...
├── run_longExp.py      # 长期预测实验入口脚本
├── run_prefetch.py     # 预计算脚本
├── settings.py         # 项目配置文件
├── requirements.txt    # 项目依赖列表
└── README.md           # 项目说明文档
```

# 交通强化学习路径规划系统
# Traffic Reinforcement Learning Route Planning System
(本项目为机器学习课程小组项目/This is the coursework project for the Machine Learning course)
基于 Q-Learning 与 LSTM 流量预测的城市交通路径规划系统，运行于 Google Colab 环境。

A urban traffic route planning system based on Q-Learning and LSTM flow prediction, running in Google Colab.

---

## 项目结构 / Project Structure

| 文件 / File | 角色 / Role | 职责 / Responsibility |
|------|------|------|
| `LSTM_Predict_System.ipynb` | **服务层 / Service Layer** | 读取路口交通数据 → 训练 LSTM → 输出未来 1 小时各方向流量预测 / Read intersection traffic data → Train LSTM → Output 1-hour directional flow predictions |
| `Reinforcement_Learning_System.ipynb` | **逻辑层 / Logic Layer** | Q-Learning 路径规划，将 LSTM 预测结果接入奖励函数 / Q-Learning route planning with LSTM predictions integrated into reward function |

两个笔记本单向调用：RL 层精准提取并调用预测层的接口函数，不触发预测层的完整训练流程。

The two notebooks have a one-way dependency: the RL layer precisely extracts and calls the prediction layer's interface function without triggering its full training pipeline.

---

## 文件路径配置 / File Path Configuration

### 预测层 / Prediction Layer (`LSTM_Predict_System.ipynb`)

```
数据目录 / Data directory:   /content/road/
文件格式 / File format:      *.xlsx（每个路口一个文件 / one file per intersection）
```

### 强化学习层 / RL Layer (`Reinforcement_Learning_System.ipynb`)

```
数据目录 / Data directory:   /content/drive/MyDrive/Docs/
预测笔记本路径 / Predict notebook path:
    /content/drive/MyDrive/Docs/LSTM_Predict_System.ipynb
文件格式 / File format:      *.xlsx（与预测层同一批数据 / same dataset as prediction layer）
```

> **注意 / Note:** 若笔记本文件名含中文或特殊字符，建议使用 `os.listdir()` 模式匹配定位文件，避免路径编码问题。
> If notebook filenames contain non-ASCII characters (Chinese/special chars), use `os.listdir()` pattern matching to locate files instead of hardcoding paths to avoid encoding issues.

---

## 系统架构 / System Architecture

```
Reinforcement_Learning_System.ipynb
│
├── Cell 0   导入库 + 挂载 Google Drive
│            Import libraries + Mount Google Drive
├── Cell 1   读取所有路口 .xlsx 文件路径
│            Load all intersection .xlsx file paths
│
├── Cell 3   环境定义 / Environment Definition
│            ├── get_state_index()      车辆数 → 拥堵等级（0~20级）/ Vehicle count → Congestion level (0–20)
│            └── Table 类 / Table class  Q-Table 环境 / Q-Table environment
│                ├── reset()            初始化 agent 位置与目标 / Initialize agent position and goal
│                ├── step()             执行动作，返回 next_state / reward / done / Execute action, return next_state / reward / done
│                ├── penalty_dis()      距离惩罚（曼哈顿距离 × 权重）/ Distance penalty (Manhattan distance × weight)
│                └── penalty_cong()     拥堵惩罚（基于 LSTM 预测的多步滚动计算）/ Congestion penalty (multi-step rolling calc based on LSTM predictions)
│
├── Cell 5   _load_function_from_notebook()
│            └── 从 LSTM_Predict_System.ipynb 精准提取 get_directional_predictions()
│                Precisely extract get_directional_predictions() from LSTM_Predict_System.ipynb
│
├── Cell 6   调用预测服务 / Call prediction service
│            └── lstm_by_intersection = get_directional_predictions(data_dir, ...)
│                返回 / Returns {路口名 / intersection_name: ndarray(12, 4)}
│
├── Cell 8   数据处理 / Data Processing
│            ├── process_intersection()  单路口统计量计算 / Single intersection statistics
│            └── 构建训练用 numpy 数组 / Build training numpy arrays
│                ├── lstm_lst   (12, 4, 12)  未来流量预测 / Future flow predictions
│                ├── mean_lst   (12, 4)      历史均值 / Historical mean
│                ├── var_lst    (12, 4)      历史方差 / Historical variance
│                └── queue_lst  (12, 4)      当前队列长度 / Current queue length
│
├── Cell 10  算法与可视化函数定义 / Algorithm & Visualization Functions
│            ├── q_learning_train()              Q-Learning 主训练循环 / Main training loop
│            ├── plot_rewards()                  奖励曲线（Y 轴固定 -200~200）/ Reward curve (Y-axis fixed -200–200)
│            ├── compute_greedy_trajectory()     贪婪策略路径推演 / Greedy policy trajectory
│            └── animate_learned_policy_pretty() 路径动画 / Path animation
│
├── Cell 12  主执行（Part 5）/ Main Execution (Part 5)
│            ├── 构建 3×4 路网环境 / Build 3×4 road network environment
│            ├── 运行 q_learning_train() / Run q_learning_train()
│            ├── 绘制奖励曲线 / Plot reward curve
│            └── 播放路径动画 / Play path animation
│
└── Cell 14  学习率敏感性分析（Part 6）/ Learning Rate Sensitivity Analysis (Part 6)
             ├── 每个学习率重复 200 次实验 / 200 repeated experiments per learning rate
             ├── 统计各路径出现频率 / Count path occurrence frequency
             └── 独立输出 Top 10 路径竖状图 / Output Top 10 path bar chart per learning rate
```

---

## 路网拓扑 / Road Network Topology

本系统以 3×4 网格模拟城市路网，共 12 个路口节点。

This system simulates an urban road network as a 3×4 grid with 12 intersection nodes.

```
 0 |  1 |  2 |  3
———+————+————+———
 4 |  5 |  6 |  7
———+————+————+———
 8 |  9 | 10 | 11
```

默认起点 / Default start：节点 0 / Node 4（第一行左端 / first row left）
默认终点 / Default goal：节点 11 / Node 11（右下角 / bottom-right）

agent 每步可向上、下、左、右移动，越界或逆向行驶会触发惩罚。

The agent can move up, down, left, or right each step; out-of-bounds or wrong-direction moves trigger penalties.

---

## 数据说明 / Data Description

- **来源 / Source**：12 个交叉口的车辆过车记录，存储为 `.xlsx` 文件 / Vehicle passage records from 12 intersections stored as `.xlsx` files
- **预测层路径 / Predict layer path**：`/content/road/`
- **RL 层路径 / RL layer path**：`/content/drive/MyDrive/Docs/`
- **字段 / Fields**：`时间` (datetime)、`方向` (1~4 / direction)、`车牌号` (vehicle plate)
- **重采样 / Resampling**：按 5 分钟粒度统计各方向过车数量 / Count vehicles per direction per 5-minute interval

---

## 奖励函数 / Reward Function

每一步奖励由三部分组成。Each step reward consists of three components.

| 情形 / Situation | 奖励值 / Reward |
|------|--------|
| 到达目标节点 / Reach goal node | `+200`（terminal） |
| 越出地图边界 / Out of bounds | `-10`（boundary） |
| 逆行（违反方向限制）/ Wrong direction | `-20`（against） |
| 正常移动 / Normal move | `−拥堵惩罚 − 距离惩罚` / `−congestion penalty − distance penalty` |

**拥堵惩罚 / Congestion Penalty** 由 `penalty_cong()` 计算，以当前队列长度为初始值，结合 LSTM 预测的未来多步流量，用折扣因子 `beta` 逐步衰减累加。

Computed by `penalty_cong()`: starts from current queue length, accumulates future LSTM-predicted flows with discount factor `beta`.

**距离惩罚 / Distance Penalty** 由 `penalty_dis()` 计算，为当前位置到目标的曼哈顿距离乘以权重 `alpha`。

Computed by `penalty_dis()`: Manhattan distance from current position to goal multiplied by weight `alpha`.

---

## 关键超参数 / Key Hyperparameters

| 参数 / Parameter | 默认值 / Default | 说明 / Description |
|------|--------|------|
| `Episodes` | 500 | 训练轮数 / Training episodes |
| `Max_steps` | 100 | 每轮最大步数 / Max steps per episode |
| `Dist_Weight` (α) | 0.1 | 距离惩罚权重 / Distance penalty weight |
| `Cong_Weight` (β) | 0.5 | 未来拥堵折扣因子 / Future congestion discount factor |
| `Dec_Weight` (γ) | 0.95 | Q-Learning 折扣因子 / Q-Learning discount factor |
| `Learning_Rate` (λ) | 0.1 | Q-Table 学习率 / Q-Table learning rate |
| `Epsilon_start` | 0.5 | 初始探索率 / Initial exploration rate |
| `Epsilon_min` | 0.05 | 最低探索率 / Minimum exploration rate |
| `Epsilon_decay` | 0.99 | 探索率衰减系数 / Exploration rate decay |

---

## LSTM 预测接口 / LSTM Prediction Interface

`LSTM_Predict_System.ipynb` 的 Cell 0 暴露以下公共函数 / Cell 0 exposes the following public function:

```python
get_directional_predictions(
    data_dir: str,                                  # 数据目录 / Data directory: /content/road/
    prediction_start_time: str = '2024-04-03 18:00:00',
    look_back: int = 12
) -> dict
```

**返回值结构 / Return Value Structure：**

```python
{
    "predicted_df"    : pd.DataFrame,           # MultiIndex 列 / columns: (路口名 / intersection, 方向 / direction)
    "by_intersection" : dict[str, np.ndarray],  # 路口名 → shape (12, 4) / intersection → shape (12, 4)
    "intersections"   : list[str],              # 12 个路口名 / 12 intersection names
    "prediction_times": pd.DatetimeIndex,       # 未来 12 个时间戳 / Next 12 timestamps
}
```

RL 笔记本通过 `_load_function_from_notebook()` 仅提取并执行该函数所在单元格，不触发 LSTM 完整训练流程。

The RL notebook uses `_load_function_from_notebook()` to extract and execute only the function cell, without triggering the full LSTM training pipeline.

---

## 学习率敏感性分析 / Learning Rate Sensitivity Analysis（Part 6）

对三个学习率（0.1、0.01、0.001）各重复 200 次独立训练实验，统计最后一个 episode 中 agent 走出的路径分布。

200 independent training experiments per learning rate (0.1, 0.01, 0.001), recording the path distribution of the agent's final episode.

**输出 / Output：**
- 控制台 / Console：完整路径频率表（含 NULL）/ Full path frequency table (including NULL)
- 图表 / Chart：每个学习率独立输出一张竖状图，展示出现次数最多的 Top 10 路径，柱顶标注次数与百分比 / One bar chart per learning rate showing Top 10 paths with count and percentage labels

**NULL 的含义 / NULL meaning：** agent 在最后一个 episode 未能到达目标节点，即学习失败的一次实验。/ The agent failed to reach the goal in the final episode, indicating a failed learning run.

---

## 运行方式 / How to Run

1. 将所有 `.xlsx` 交通数据文件上传至 Google Drive / Upload all `.xlsx` traffic data files to Google Drive:
   - 预测层数据 / Predict layer data → `MyDrive/road/`（或按需修改 `data_dir` / or modify `data_dir` as needed）
   - RL 层数据 / RL layer data → `MyDrive/Docs/`
2. 将两个 `.ipynb` 文件上传至 `MyDrive/Docs/` / Upload both `.ipynb` files to `MyDrive/Docs/`
3. 按顺序运行 `Reinforcement_Learning_System.ipynb` 的各单元格 / Run cells of `Reinforcement_Learning_System.ipynb` in order
   - Cell 5/6 会自动调用 `LSTM_Predict_System.ipynb` 获取预测结果 / Cell 5/6 automatically calls `LSTM_Predict_System.ipynb` for predictions
   - 无需手动运行预测笔记本 / No need to run the prediction notebook manually
4. Cell 12 运行完毕后输出奖励曲线和路径动画 / Cell 12 outputs reward curve and path animation
5. Cell 14 运行完毕后输出学习率敏感性分析图（约需数分钟）/ Cell 14 outputs sensitivity analysis charts (takes a few minutes)

---

## 依赖环境 / Dependencies

```
Python 3.x（Google Colab 默认 / default）
tensorflow / keras
numpy
pandas
matplotlib
scikit-learn
openpyxl
```

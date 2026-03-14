# DiscreteNetmork_main.py 代码逐行解释

> 本文档按主函数的运行顺序，逐段解释 `DiscreteNetmork_main.py` 中的每一行代码。涉及计算的部分均附有对应的数学公式，行号在前，公式在后。

---

## 一、环境准备与依赖导入（第 1–30 行）

```python
1.  #!/usr/bin/env python
2.  # coding: utf-8
```

- **第 1 行**：Shebang 行，声明使用系统 Python 解释器运行脚本。
- **第 2 行**：声明源文件编码为 UTF-8，保证中文等非 ASCII 字符可正常处理。

```python
8.  get_ipython().system('pip install optuna')
9.  import optuna
10. import torch
11. import random
12. import torch.nn as nn
13. import numpy as np
14. import matplotlib.pyplot as plt
15. import matplotlib
16. import torch.nn.functional as F
17. from csv import writer
18. import seaborn as sns
19. import os
20. os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

- **第 8 行**：在 Jupyter Notebook 环境中执行 shell 命令，安装 `optuna` 超参数优化库。
- **第 9–19 行**：导入所有必要的第三方库：
  - `optuna`：贝叶斯超参数优化框架。
  - `torch`：PyTorch 深度学习框架。
  - `random`：Python 标准随机数库（用于可重复性）。
  - `torch.nn`：PyTorch 神经网络模块。
  - `numpy`：数值计算库。
  - `matplotlib.pyplot`、`matplotlib`：绘图库。
  - `torch.nn.functional`：PyTorch 函数式神经网络接口（含激活函数）。
  - `csv.writer`：CSV 文件写入工具。
  - `seaborn`：基于 matplotlib 的统计可视化库。
  - `os`：操作系统接口。
- **第 20 行**：设置环境变量，解决 Intel MKL 与其他数学库并存时可能出现的动态链接库冲突。

```python
26. from Scripts.GetData import getDataLoaders, loadData
27. from Scripts.Training import train
28. from Scripts.PlotResults import plotResults
29. from Scripts.SavedParameters import hyperparams
30. import pandas as pd
```

- **第 26–29 行**：从本地 `Scripts` 包导入自定义模块：
  - `getDataLoaders`：构建训练/验证/测试数据加载器。
  - `loadData`：读取原始数据并返回节点信息。
  - `train`：执行神经网络训练循环。
  - `plotResults`：绘制预测结果与真实值的对比图。
  - `hyperparams`：根据数据案例和训练比例返回预存的最优超参数字典。
- **第 30 行**：导入 `pandas`，用于数据处理（间接依赖）。

---

## 二、绘图参数配置（第 37–42 行）

```python
37. sns.set_style("darkgrid")
38. sns.set(font = "Times New Roman")
39. sns.set_context("paper")
40. plt.rcParams['mathtext.fontset'] = 'cm'
41. plt.rcParams['font.family'] = 'STIXGeneral'
42. plt_kws = {"rasterized": True}
```

- **第 37 行**：将 seaborn 风格设为深色网格背景。
- **第 38 行**：全局字体设为 Times New Roman（学术论文常用衬线字体）。
- **第 39 行**：将绘图上下文设为 `"paper"`，缩小元素尺寸以适配期刊图幅。
- **第 40–41 行**：配置 matplotlib 数学文本使用 Computer Modern 字体集，正文字体使用 STIXGeneral，与 LaTeX 风格保持一致。
- **第 42 行**：定义关键字参数字典 `plt_kws`，在生成大量散点/曲线时开启栅格化以减小文件体积。

---

## 三、全局精度与设备配置（第 48–55 行）

```python
48. torch.set_default_dtype(torch.float32)
```

- **第 48 行**：将 PyTorch 全局默认张量类型设为单精度浮点数（float32），在精度与计算效率之间取得平衡。

```python
54. device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
55. print(device)
```

- **第 54–55 行**：自动检测是否有可用 GPU。若有，则使用编号为 0 的 CUDA 设备以加速训练；否则退而使用 CPU。

---

## 四、用户输入与随机种子固定（第 61–68 行）

```python
61. datacase = int(input("Which datacase do you want to work with?\n"))
62. percentage_train = float(input("Which percentage of the dataset do you want to use for training? Choose among 0.1,0.2,0.4,0.8\n"))
64. print(f"\n\n Case with percentage_train={percentage_train} and datacase={datacase}\n\n")
66. torch.manual_seed(1)
67. np.random.seed(1)
68. random.seed(1)
```

- **第 61 行**：读取用户输入的数据案例编号（`datacase`），决定使用哪组边界条件数据（如 `1` 表示两端均给定边界条件，`2` 表示仅右端给定边界条件）。
- **第 62 行**：读取用于训练的数据集比例（`percentage_train`），可选 0.1、0.2、0.4、0.8。
- **第 64 行**：打印当前运行配置，便于日志追踪。
- **第 66–68 行**：分别为 PyTorch、NumPy 和 Python 内置 `random` 模块固定随机种子为 1，保证实验结果的可重复性。

---

## 五、`approximate_curve` 神经网络类（第 70–102 行）

### 5.1 类定义与 `__init__` 方法（第 70–88 行）

```python
70. class approximate_curve(nn.Module):
71.     def __init__(self, is_res=True, normalize=True, act_name='tanh',
                    nlayers=3, hidden_nodes=50, output_dim=204):
72.         super().__init__()
74.         torch.manual_seed(1)
75.         np.random.seed(1)
76.         random.seed(1)
77.         self.act_dict = {"tanh":   lambda x: torch.tanh(x),
78.                          "sigmoid": lambda x: torch.sigmoid(x),
79.                          "swish":   lambda x: x * torch.sigmoid(x),
80.                          "relu":    lambda x: torch.relu(x),
81.                          "lrelu":   lambda x: F.leaky_relu(x)}
82.         self.is_norm = normalize
83.         self.is_res  = is_res
84.         self.act     = self.act_dict[act_name]
85.         self.nlayers = nlayers
86.         self.first   = nn.Linear(8, hidden_nodes)
87.         self.linears = nn.ModuleList([nn.Linear(hidden_nodes, hidden_nodes)
                                         for i in range(self.nlayers)])
88.         self.last    = nn.Linear(hidden_nodes, output_dim)
```

- **第 70 行**：定义 `approximate_curve` 类，继承自 `nn.Module`，是 PyTorch 所有神经网络模型的基类。
- **第 71 行**：构造函数，参数含义：
  - `is_res`：是否使用残差连接（ResNet 风格）。
  - `normalize`：是否对输入进行归一化。
  - `act_name`：激活函数名称。
  - `nlayers`：隐藏层数量。
  - `hidden_nodes`：每个隐藏层的神经元数。
  - `output_dim`：输出维度，等于 $4(N-2)$，其中 $N$ 为网格节点总数。
- **第 72 行**：调用父类初始化，注册参数与子模块。
- **第 74–76 行**：在模型初始化时再次固定随机种子，确保每次构建模型时权重初始化一致。
- **第 77–81 行**：构建激活函数字典，可选：
  - `tanh`：双曲正切，$\sigma(x) = \tanh(x)$；
  - `sigmoid`：$\sigma(x) = \dfrac{1}{1+e^{-x}}$；
  - `swish`：$\sigma(x) = x \cdot \dfrac{1}{1+e^{-x}}$；
  - `relu`：$\sigma(x) = \max(0, x)$；
  - `lrelu`（Leaky ReLU）：$\sigma(x) = \max(\alpha x, x)$，$\alpha$ 为小正数（默认 0.01）。
- **第 82–85 行**：保存归一化开关、残差开关、激活函数选择和层数到实例属性。
- **第 86 行**：定义第一层全连接层，将 8 维输入映射到 `hidden_nodes` 维隐空间：

$$W^{(1)} \in \mathbb{R}^{H \times 8}, \quad b^{(1)} \in \mathbb{R}^{H}$$

- **第 87 行**：构建 `nlayers` 个隐藏层组成的 `ModuleList`，每层均为 $H \to H$ 的全连接层：

$$W^{(l)} \in \mathbb{R}^{H \times H}, \quad b^{(l)} \in \mathbb{R}^{H}, \quad l = 1, \ldots, L$$

- **第 88 行**：定义输出层，将隐空间映射到 $4(N-2)$ 维输出：

$$W^{(\text{out})} \in \mathbb{R}^{4(N-2) \times H}, \quad b^{(\text{out})} \in \mathbb{R}^{4(N-2)}$$

---

### 5.2 `forward` 前向传播方法（第 90–102 行）

```python
90.  def forward(self, x):
92.      if self.is_norm:
93.          x[:,0] = (x[:,0] - 1.5) / 1.5
94.          x[:,4] = (x[:,4] - 1.5) / 1.5
95.      x = self.act(self.first(x))
96.      for i in range(self.nlayers):
97.          if self.is_res:
98.              x = x + self.act(self.linears[i](x))
99.          else:
100.             x = self.act(self.linears[i](x))
102.     return self.last(x)
```

- **第 92–94 行**：若 `is_norm=True`，对输入 $\mathbf{x}$ 的第 0 列和第 4 列（对应两端边界处的某特定物理量）做 Min-Max 归一化，使其缩放到 $[-1, 1]$ 区间：

$$\tilde{x}_i = \frac{x_i - 1.5}{1.5}, \quad i \in \{0, 4\}$$

- **第 95 行**：通过第一层全连接加激活函数，将归一化后的 8 维输入 $\tilde{\mathbf{x}}$ 映射到 $H$ 维隐向量 $\mathbf{h}^{(0)}$：

$$\mathbf{h}^{(0)} = \sigma\!\left(W^{(1)} \tilde{\mathbf{x}} + \mathbf{b}^{(1)}\right)$$

- **第 96–100 行**：逐层通过 $L$ 个隐藏层。有两种模式：

  **普通 MLP（`is_res=False`）**（第 100 行）：

  $$\mathbf{h}^{(l+1)} = \sigma\!\left(W^{(l+1)} \mathbf{h}^{(l)} + \mathbf{b}^{(l+1)}\right), \quad l = 0, 1, \ldots, L-1$$

  **残差网络（`is_res=True`，第 98 行）**：在普通前向计算的基础上加上恒等跳跃连接，缓解深层网络梯度消失问题：

  $$\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \sigma\!\left(W^{(l+1)} \mathbf{h}^{(l)} + \mathbf{b}^{(l+1)}\right), \quad l = 0, 1, \ldots, L-1$$

- **第 102 行**：通过输出层（线性，无激活函数）将最终隐向量 $\mathbf{h}^{(L)}$ 映射为预测的离散曲线坐标向量：

$$\hat{\mathbf{y}} = W^{(\text{out})} \mathbf{h}^{(L)} + \mathbf{b}^{(\text{out})}, \quad \hat{\mathbf{y}} \in \mathbb{R}^{4(N-2)}$$

  其中 $\hat{\mathbf{y}}$ 包含网格内部 $N-2$ 个节点处的 4 个物理量（例如位置 $x$、$y$ 及两个切向分量）的预测值。整个网络的作用可概括为：给定 8 维边界条件 $\boldsymbol{\chi}$，输出对离散曲线内节点状态的近似：

$$\hat{\mathbf{u}}_b \approx \mathcal{N}_\theta(\boldsymbol{\chi})$$

---

## 六、加载数据节点信息（第 104 行）

```python
104. num_nodes, _, _ = loadData(datacase)
```

- **第 104 行**：调用 `loadData` 函数，传入数据案例编号，返回网格节点数 `num_nodes`（以及其他被忽略的返回值）。`num_nodes` 决定了网络输出维度 $4(N-2)$，即内部节点数乘以每节点的状态变量数。

---

## 七、`define_model` 超参数搜索模型构建函数（第 106–117 行）

```python
106. def define_model(trial):
107.     torch.manual_seed(1)
108.     np.random.seed(1)
109.     random.seed(1)
110.     is_res = False
111.     normalize = True
112.     act_name = "tanh"
113.     nlayers = trial.suggest_int("n_layers", 0, 10)
114.     hidden_nodes = trial.suggest_int("hidden_nodes", 10, 1000)
115.     model = approximate_curve(is_res, normalize, act_name, nlayers, hidden_nodes,
                                   output_dim=int(4*(num_nodes-2)))
117.     return model
```

- **第 106 行**：定义模型构建函数，接受 Optuna `trial` 对象，用于超参数搜索时按 trial 建立不同配置的模型。
- **第 107–109 行**：每次构建新模型前固定随机种子，确保不同 trial 的唯一差异来自超参数而非随机初始化。
- **第 110–112 行**：固定超参数：不使用残差连接、开启输入归一化、激活函数为 `tanh`。
- **第 113 行**：由 Optuna 在整数范围 $[0, 10]$ 内搜索隐藏层数量 `n_layers`：

$$L \in \{0, 1, 2, \ldots, 10\}$$

- **第 114 行**：由 Optuna 在整数范围 $[10, 1000]$ 内搜索每层神经元数 `hidden_nodes`：

$$H \in \{10, 11, \ldots, 1000\}$$

- **第 115–116 行**：用搜索到的超参数实例化 `approximate_curve` 模型，输出维度固定为 $4(N-2)$，并返回该模型。

---

## 八、`objective` Optuna 目标函数（第 121–185 行）

```python
121. def objective(trial):
123.     torch.manual_seed(1)
124.     np.random.seed(1)
125.     random.seed(1)
128.     model = define_model(trial)
129.     model.to(device)
131.     lr = 1e-3
132.     weight_decay = 0
133.     gamma = trial.suggest_float("gamma", 0, 1e-2)
134.     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
136.     criterion = nn.MSELoss()
138.     batch_size = 32
139.     _, _, _, _, x_val, y_val, trainloader, _, valloader = getDataLoaders(
                batch_size, datacase, percentage_train)
```

- **第 121 行**：定义 Optuna 目标函数，Optuna 将最小化此函数的返回值（验证集 MSE）。
- **第 123–125 行**：固定随机种子。
- **第 128–129 行**：构建当前 trial 对应的模型并迁移到目标设备。
- **第 131–132 行**：固定学习率 $\eta = 10^{-3}$ 和权重衰减系数 $\lambda = 0$。
- **第 133 行**：由 Optuna 在连续区间 $[0, 10^{-2}]$ 内搜索学习率调度衰减系数 `gamma`：

$$\gamma \in [0,\, 10^{-2}]$$

- **第 134 行**：实例化 Adam 优化器，其参数更新规则为：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

  其中 $g_t$ 为当前梯度，$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$（PyTorch 默认值），$\hat{m}_t = m_t / (1-\beta_1^t)$，$\hat{v}_t = v_t / (1-\beta_2^t)$ 为偏差修正项。

- **第 136 行**：实例化均方误差损失函数（MSELoss）：

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left\| \hat{\mathbf{y}}_i - \mathbf{y}_i \right\|^2$$

  其中 $\hat{\mathbf{y}}_i = \mathcal{N}_\theta(\mathbf{x}_i)$ 为模型预测，$\mathbf{y}_i$ 为真实离散曲线状态向量。

- **第 138–139 行**：设置批大小为 32，调用 `getDataLoaders` 获取验证集输入 `x_val`、验证集标签 `y_val`、训练数据加载器 `trainloader` 和验证数据加载器 `valloader`。

```python
141.     print("Current test with :\n\n")
142.     for key, value in trial.params.items():
143.         print("    {}: {}".format(key, value))
146.     epochs = 300
147.     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                    step_size=int(0.45*epochs), gamma=0.1)
149.     loss = train(model, gamma, criterion, scheduler, optimizer, epochs,
                     trainloader, valloader, device)
150.     print('Loss ', loss.item())
151.     error = 1000
```

- **第 141–143 行**：打印当前 trial 的超参数组合，便于监控。
- **第 146 行**：设置最大训练轮数 $E = 300$。
- **第 147 行**：实例化步进学习率调度器（StepLR）。每 $\lfloor 0.45 \times 300 \rfloor = 135$ 个 epoch 将学习率乘以 0.1：

$$\eta_t = \eta_0 \cdot \gamma_{\text{step}}^{\lfloor t / T_{\text{step}} \rfloor}, \quad T_{\text{step}} = 135, \quad \gamma_{\text{step}} = 0.1$$

- **第 149 行**：调用 `train` 函数，执行完整的训练循环（前向传播、损失计算、反向传播、参数更新），并在每轮结束后调用调度器更新学习率。返回最终训练损失。
- **第 151 行**：初始化验证误差为 1000（代表未评估时的极大值，若训练发散则保持此默认值）。

```python
153.     if not torch.isnan(loss):
154.         model.eval()
156.         learned_traj = np.zeros_like(y_val)
158.         bcs_val = torch.from_numpy(x_val.astype(np.float32)).to(device)
159.         learned_traj = model(bcs_val).detach().cpu().numpy()
160.         error = np.mean((learned_traj - y_val)**2)
162.         print(f"The error on the validation trajectories is: {error}.")
```

- **第 153 行**：检查训练损失是否为 NaN（非数），若训练发散则跳过评估，保留 `error=1000`。
- **第 154 行**：将模型切换到推理模式，关闭 Dropout/BatchNorm 的训练行为。
- **第 156–159 行**：将验证集输入转为 float32 张量并迁移到 device，通过模型前向传播得到预测的轨迹；`.detach()` 断开计算图，`.cpu().numpy()` 转回 NumPy 数组。
- **第 160 行**：计算验证集均方误差（MSE）：

$$\text{MSE}_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} \left\| \hat{\mathbf{y}}_i^{(\text{val})} - \mathbf{y}_i^{(\text{val})} \right\|^2$$

```python
165.     if trial.number == 0:
166.         labels = []
167.         for lab, _ in trial.params.items():
168.             labels.append(str(lab))
169.         labels.append("MSE")
170.         with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
171.             writer_object = writer(f_object)
172.             writer_object.writerow(labels)
173.             f_object.close()
175.     results = []
176.     for _, value in trial.params.items():
177.         results.append(str(value))
179.     results.append(error)
181.     with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
182.         writer_object = writer(f_object)
183.         writer_object.writerow(results)
184.         f_object.close()
185.     return error
```

- **第 165–173 行**：仅在第一个 trial（`trial.number == 0`）时写入 CSV 表头（各超参数名称 + `"MSE"`），避免重复写头行。
- **第 175–184 行**：将当前 trial 的超参数值和对应的验证 MSE 追加写入 CSV 文件，文件名包含训练比例百分比（如 `results10_Fig2.csv`）。
- **第 185 行**：返回验证集 MSE 作为 Optuna 的优化目标（越小越好）。

---

## 九、Optuna 超参数搜索（第 187–198 行）

```python
187. optuna_study = input("Do you want to do hyperparameter test? Type yes or no: ")
188. params = {}
189. if optuna_study == "yes":
190.     optuna_study = True
191. else:
192.     optuna_study = False
193. if optuna_study:
194.     study = optuna.create_study(direction="minimize", study_name="Euler Elastica")
195.     study.optimize(objective, n_trials=5)
196.     print("Study statistics: ")
197.     print("Number of finished trials: ", len(study.trials))
198.     params = study.best_params
```

- **第 187–192 行**：询问用户是否启动超参数搜索，将字符串回答转换为布尔值。
- **第 194 行**：创建 Optuna study 对象，目标方向为最小化（`direction="minimize"`），研究名称为 `"Euler Elastica"`（欧拉弹性曲线，揭示了该网络所学习的物理问题背景）。
- **第 195 行**：执行 5 次 trial（`n_trials=5`），每次 trial 由 Optuna 的 TPE（树形 Parzen 估计）贝叶斯采样算法提议新的超参数组合，以近似最小化：

$$\theta^* = \arg\min_{\theta \in \Theta} \text{MSE}_{\text{val}}(\theta)$$

  其中 $\Theta = \{L \in [0,10]\} \times \{H \in [10,1000]\} \times \{\gamma \in [0, 10^{-2}]\}$。

- **第 196–198 行**：打印完成的 trial 数量，并将最佳超参数字典赋给 `params`。

---

## 十、超参数选取逻辑（第 200–221 行）

```python
200. torch.manual_seed(1)
201. np.random.seed(1)
202. random.seed(1)
204. manual_input = False
205. if params == {}:
207.     if manual_input:
208.         print("No parameters have been specified. Let's input them:\n\n")
209.         nlayers      = int(input("How many layers do you want the network to have? "))
210.         hidden_nodes = int(input("How many hidden nodes do you want the network to have? "))
211.         weight_decay = float(input("What weight decay do you want to use? "))
212.         gamma        = float(input("What value do you want for gamma? "))
213.         batch_size   = int(input("What batch size do you want? "))
215.         params = {'n_layers': nlayers, 'hidden_nodes': hidden_nodes, 'gamma': gamma}
218.     else:
220.         params = hyperparams(datacase, percentage_train)
221. print(f'The hyperparameters yelding the best results for this case are: {params}')
```

- **第 200–202 行**：再次固定随机种子，保持后续模型构建的一致性。
- **第 204 行**：`manual_input = False`，默认不启用手动输入模式（此开关供开发者调试用）。
- **第 205 行**：若 `params` 为空字典（即未进行 Optuna 搜索），则需要指定超参数。
- **第 207–215 行**：`manual_input=True` 时，通过交互式输入手动指定 `nlayers`、`hidden_nodes`、`weight_decay`、`gamma`、`batch_size` 并构建 `params` 字典。
- **第 220 行**：`manual_input=False`（默认）时，调用 `hyperparams(datacase, percentage_train)` 从预存最优参数表中查表，返回该数据案例和训练比例下已知最优的超参数字典。
- **第 221 行**：打印最终使用的超参数。

---

## 十一、`define_best_model` 最优模型构建函数（第 222–239 行）

```python
222. def define_best_model():
224.     torch.manual_seed(1)
225.     np.random.seed(1)
226.     random.seed(1)
228.     normalize    = True
229.     act          = "tanh"
230.     nlayers      = params["n_layers"]
231.     hidden_nodes = params["hidden_nodes"]
232.     is_res       = False
234.     print("Nodes: ", hidden_nodes)
236.     model = approximate_curve(is_res, normalize, act, nlayers, hidden_nodes,
                                   int(4*(num_nodes-2)))
238.     return model
239. model = define_best_model()
240. model.to(device)
```

- **第 222 行**：定义最优模型构建函数，使用前面确定的最优超参数字典 `params`（来自 Optuna、手动输入或预存表）。
- **第 224–226 行**：固定随机种子。
- **第 228–232 行**：从 `params` 中提取层数和每层神经元数，固定使用 MLP（`is_res=False`）、输入归一化和 `tanh` 激活函数。
- **第 236–237 行**：实例化最优配置的 `approximate_curve` 模型，输出维度为 $4(N-2)$。
- **第 239–240 行**：调用 `define_best_model()` 构建模型实例，并迁移到目标设备（GPU 或 CPU）。

---

## 十二、训练配置参数（第 242–254 行）

```python
242. TrainMode   = input("Train Mode True or False? Type 0 for False and 1 for True: ") == "1"
243. weight_decay = 0.
244. lr           = 1e-3
245. gamma        = params["gamma"]
246. nlayers      = params["n_layers"]
247. hidden_nodes = params["hidden_nodes"]
248. batch_size   = 32
249. epochs       = 300
250. optimizer    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
251. scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.45*epochs),
                                                    gamma=0.1)
252. criterion    = nn.MSELoss()
253. x_train, y_train, x_test, y_test, x_val, y_val, trainloader, testloader, valloader = \
         getDataLoaders(batch_size, datacase, percentage_train)
254. model.to(device)
```

- **第 242 行**：询问用户是否重新训练模型（`1` = 训练，`0` = 加载预训练权重）。
- **第 243–248 行**：设置完整训练配置：
  - 权重衰减 $\lambda = 0$（不使用 L2 正则化）；
  - 初始学习率 $\eta_0 = 10^{-3}$；
  - 从 `params` 中读取调度衰减系数 $\gamma$、层数和节点数；
  - 批大小 $B = 32$，最大训练轮数 $E = 300$。
- **第 250 行**：实例化 Adam 优化器（更新规则见第 134 行注释）。
- **第 251 行**：实例化 StepLR 调度器：每 $\lfloor 0.45 \times 300 \rfloor = 135$ 轮将学习率乘以固定步进衰减系数 $\gamma_{\text{step}} = 0.1$（注意：此处的 `gamma=0.1` 是**固定的步进衰减系数**，与 Optuna 搜索得到的超参数 `gamma`（第 133/245 行）含义完全不同，后者作为 `train` 函数的正则化或权重缩放系数传入）：

$$\eta_t = \eta_0 \times \gamma_{\text{step}}^{\lfloor t / T_{\text{step}} \rfloor}, \quad \gamma_{\text{step}} = 0.1, \quad T_{\text{step}} = 135$$

- **第 252 行**：实例化 MSELoss：

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left\| \mathcal{N}_\theta(\mathbf{x}_i) - \mathbf{y}_i \right\|^2$$

- **第 253 行**：调用 `getDataLoaders` 获取完整数据集分割：训练集 $(\mathbf{x}_{\text{train}}, \mathbf{y}_{\text{train}})$、测试集 $(\mathbf{x}_{\text{test}}, \mathbf{y}_{\text{test}})$、验证集 $(\mathbf{x}_{\text{val}}, \mathbf{y}_{\text{val}})$ 及对应的 DataLoader。

---

## 十三、训练或加载预训练模型（第 256–268 行）

```python
256. if TrainMode:
257.     loss = train(model, gamma, criterion, scheduler, optimizer, epochs,
                     trainloader, valloader, device)
258.     if datacase == 1:
259.         torch.save(model.state_dict(),
                        f'TrainedModels/BothEnds{percentage_train}data.pt')
260.     if datacase == 2:
261.         torch.save(model.state_dict(),
                        f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt')
262. else:
263.     if datacase == 1:
264.         pretrained_dict = torch.load(
                        f'TrainedModels/BothEnds{percentage_train}data.pt',
                        map_location=device)
265.     if datacase == 2:
266.         pretrained_dict = torch.load(
                        f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt',
                        map_location=device)
267.     model.load_state_dict(pretrained_dict)
268. model.eval()
```

- **第 256–257 行**：若 `TrainMode=True`，调用 `train` 函数执行正式训练，其核心计算流程为：

  对于每个批次 $\mathcal{B} \subset \{1, \ldots, n\}$，执行：

  1. **前向传播**：$\hat{\mathbf{Y}}_\mathcal{B} = \mathcal{N}_\theta(\mathbf{X}_\mathcal{B})$
  2. **损失计算**：$\mathcal{L}_\mathcal{B} = \dfrac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|^2$（批 MSE）
  3. **反向传播**：$g_t = \nabla_\theta \mathcal{L}_\mathcal{B}$
  4. **参数更新**：Adam 步（见第 134 行公式）
  5. **学习率调度**：每 135 个 epoch 触发一次 $\eta \leftarrow 0.1\eta$

- **第 258–261 行**：训练完成后，根据数据案例将模型权重字典（`.state_dict()`）保存到对应的 `.pt` 文件：
  - `datacase=1`（两端均给定边界条件）→ `BothEnds{percentage_train}data.pt`
  - `datacase=2`（仅右端给定边界条件）→ `BothEndsRightEnd{percentage_train}data.pt`

- **第 262–267 行**：若 `TrainMode=False`，从磁盘加载对应的预训练权重字典，并通过 `model.load_state_dict` 将其赋给当前模型，实现迁移推理。`map_location=device` 保证在 CPU 环境下也能加载 GPU 训练的模型。

- **第 268 行**：无论训练还是加载，均切换到推理模式（关闭 Dropout 和 BatchNorm 的随机性）。

---

## 十四、结果评估与可视化（第 270–277 行）

```python
270. torch.manual_seed(1)
271. np.random.seed(1)
272. random.seed(1)
274. model.eval()
276. # printing the accuracies and plotting the results
277. plotResults(model, device, x_train, y_train, x_test, y_test,
                x_val, y_val, num_nodes, datacase, percentage_train,
                gamma, nlayers, hidden_nodes)
```

- **第 270–272 行**：固定随机种子，确保评估过程（如有随机采样操作）可重复。
- **第 274 行**：再次确认模型处于推理模式。
- **第 277 行**：调用 `plotResults` 函数，在训练集、测试集和验证集上评估并可视化模型性能。该函数内部通常会：
  1. 对各数据子集进行前向推理，得到预测曲线 $\hat{\mathbf{Y}}$；
  2. 计算相对误差或均方误差：$\text{err}_{\text{rel}} = \dfrac{\|\hat{\mathbf{Y}} - \mathbf{Y}\|_F}{\|\mathbf{Y}\|_F}$；
  3. 绘制预测轨迹与真实轨迹的对比图，并标注误差统计。

---

## 十五、推理耗时测量（第 283–291 行）

```python
283. import time
284. test_bvs    = torch.from_numpy(x_test.astype(np.float32))
285. initial_time = time.time()
286. preds        = model(test_bvs)
287. final_time   = time.time()
288. total_time   = final_time - initial_time
289. print("Number of trajectories in the test set : ", len(test_bvs))
290. print("Total time to predict test trajectories : ", total_time)
291. print("Average time to predict test trajectories : ", total_time / len(test_bvs))
```

- **第 283 行**：导入 `time` 模块，用于计时。
- **第 284 行**：将测试集输入数组 `x_test` 转换为 float32 PyTorch 张量（未迁移到 GPU，在 CPU 上测量推理时延）。
- **第 285–287 行**：记录推理前后的 Unix 时间戳，计算批量推理总耗时：

$$t_{\text{total}} = t_{\text{final}} - t_{\text{initial}}$$

- **第 288 行**：总耗时 $t_{\text{total}}$（单位：秒）。
- **第 289–291 行**：打印测试集样本数量 $N_{\text{test}}$、总推理时间和平均单次推理时间：

$$\bar{t} = \frac{t_{\text{total}}}{N_{\text{test}}}$$

  该指标量化了模型在替代传统数值求解（如有限元）时的计算加速比：神经网络的 $\bar{t}$ 通常远小于一次有限元迭代的耗时，体现了数据驱动方法在实时预测场景中的优势。

---

## 附录：网络结构总览

| 层名称 | 类型 | 输入维度 | 输出维度 | 激活函数 |
|--------|------|----------|----------|---------|
| `first` | Linear | 8 | $H$ | $\tanh$ |
| `linears[0]` ~ `linears[L-1]` | Linear (×$L$) | $H$ | $H$ | $\tanh$（+残差可选）|
| `last` | Linear | $H$ | $4(N-2)$ | 无 |

**输入**（8 维边界条件向量 $\boldsymbol{\chi}$）描述离散弹性曲线两端的几何与力学约束（位置、切向角、曲率等）。**输出**（$4(N-2)$ 维）为内部节点的完整状态向量预测 $\hat{\mathbf{u}}_b$，满足：

$$\hat{\mathbf{u}}_b = \mathcal{N}_\theta(\boldsymbol{\chi}) \approx \mathbf{u}_b^*(\boldsymbol{\chi})$$

其中 $\mathbf{u}_b^*(\boldsymbol{\chi})$ 为对应边界条件下离散欧拉弹性曲线方程的真实解。

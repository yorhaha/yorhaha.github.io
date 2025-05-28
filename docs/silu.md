---
title: SiLU
---

# SiLU

SiLU（Sigmoid-weighted Linear Unit）是一种激活函数，首次出现在2017年的论文《Self-Normalizing Neural Networks》中，后来在EfficientNet等模型中广泛应用。它结合了Sigmoid函数的平滑性和线性单元的梯度特性，表现优于传统激活函数（如ReLU）在某些任务中。

## **SiLU的定义**

SiLU的数学表达式为：

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

等价于：
$$
\text{SiLU}(x) = \frac{x}{1 + e^{-x}}
$$

## **特点与优势**

1. **平滑性**：
   - SiLU在全体实数域上连续可导（$C^\infty$平滑），而ReLU在$x=0$处不可导。
   - 梯度更稳定，有助于缓解梯度消失问题。

2. **自门控机制**：
   - 通过Sigmoid函数对输入进行加权，形成类似“开关”的效果：当$x$较大时趋近线性，较小时趋近0。
   - 这种特性类似于LSTM或GRU中的门控机制，能自适应调节信息流动。

3. **性能表现**：
   - 在深层网络中，SiLU的收敛速度和最终精度常优于ReLU和LeakyReLU。
   - 尤其适用于图像分类、目标检测等任务（如EfficientNet、YOLOv7等模型）。

## **导数计算**

SiLU的导数为：
$$
\frac{d}{dx}\text{SiLU}(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$
由于$\sigma(x)$的值在$(0,1)$之间，梯度始终非零，避免了ReLU的“神经元死亡”问题。

## **与其他激活函数的对比**

| 激活函数      | 公式                      | 特点                                                         |
| ------------- | ------------------------- | ------------------------------------------------------------ |
| **ReLU**      | $\max(0, x)$              | 计算简单，但负半轴梯度为0，可能导致神经元死亡。              |
| **LeakyReLU** | $\max(0.01x, x)$          | 负半轴引入微小斜率，缓解神经元死亡，但需手动设定超参数。     |
| **GELU**      | $x \cdot \Phi(x)$         | 类似SiLU，用高斯误差函数（CDF）加权，计算略复杂（如BERT中使用）。 |
| **Swish**     | $x \cdot \sigma(\beta x)$ | SiLU的推广形式（$\beta=1$时为SiLU），Google提出，表现优异。  |

## **代码实现示例**

```python
import torch
import numpy as np

def silu(x):
    return x * torch.sigmoid(x)  # 或 x / (1 + torch.exp(-x))

# PyTorch内置实现（torch>=1.7）
silu = torch.nn.SiLU()

# NumPy版本
def silu_np(x):
    return x / (1 + np.exp(-x))
```

## **应用场景**

- **计算机视觉**：EfficientNet、YOLOv7等模型。
- **自然语言处理**：部分Transformer变体。
- **替代ReLU**：在需要平滑梯度的深层网络中尝试SiLU可能获得更好效果。
# -*- coding: utf-8 -*-
"""DAE-TSE / KCE 里 `dae_modules.py` 没有搬过来的两个纯函数。

`dae_modules.py` 只移植了网络构建块（WordEmbedding / PositionalEncoding /
MultiHeadAtt / MultiHeadCrossAtt / DepthWiseConv / FNNBlock / TransformerLayer /
make_mask / combine_mask），下面两个是复刻 `AEDKWSASRPhone` 与其 attention 评测
时还需要的：

- `compute_dim_reduction`：源自 `kce/src/kce/model/NetModules.py:BaseConv`，
  卷积下采样后的长度公式。`AEDKWSASRPhone.forward` 对 `sph_len` 连调两次
  （kernel=3, stride=2, padding=0, dilation=1）来跟踪 `au_conv` 的时间降采样。
- `find_max_path`：源自 `kce/src/kce/eval/attention/metrics.py`，在
  (keyword_len, T) 的 cross-attention 图上做单调 DP，取最大路径和与其时间跨度，
  是关键词检测打分器的核心。

两者都逐字复刻，只补了 docstring。无 `tables.register`。
"""
# =============================================================================
# Adapted from the DAE-TSE / KCE project:
#   repo   : https://github.com/GnafiY/DAE-TSE
#   branch : cursor/cloud-agent-1784704881858-cfrvr
#   source : kce/src/kce/model/NetModules.py (BaseConv.compute_dim_reduction)
#            kce/src/kce/eval/attention/metrics.py (find_max_path)
# All credit to the original authors.
# =============================================================================
import numpy as np
import torch


def compute_dim_reduction(input_dim, kernel, stride, padding, dilation, dim=0):
    """卷积/池化后的输出长度，逐字复刻 `NetModules.BaseConv.compute_dim_reduction`。

    `input_dim` 既可以是 int，也可以是长度张量（(B,) 的 int tensor），后者会逐元素
    计算——`AEDKWSASRPhone.forward` 正是拿它算 `sph_len`。`kernel/stride/padding/
    dilation` 允许给标量或 (h, w) 二元组，给二元组时用 `dim` 选取哪一维。

    Args:
        input_dim: 输入长度，int / torch.Tensor / tuple。
        kernel: 卷积核大小，int 或序列。
        stride: 步长，int 或序列。
        padding: padding，int 或序列。
        dilation: dilation，int 或序列。
        dim: 当上面参数是序列时取第几维。

    Returns:
        与 `input_dim` 同类型的输出长度。
    """
    p = padding if isinstance(padding, int) else padding[dim]
    d = dilation if isinstance(dilation, int) else dilation[dim]
    k = kernel if isinstance(kernel, int) else kernel[dim]
    s = stride if isinstance(stride, int) else stride[dim]
    idim = input_dim if not isinstance(input_dim, tuple) else input_dim[dim]
    rdim = idim + 2 * p - d * (k - 1) - 1
    if isinstance(rdim, torch.Tensor):
        rdim = torch.div(rdim, s, rounding_mode="floor") + 1
    else:
        rdim = rdim // s + 1
    return rdim


def find_max_path(matrix):
    """在 (K, T) 矩阵上找单调递增路径的最大和，逐字复刻 KCE 的 `find_max_path`。

    路径从 `(0, t0)` 出发、每步 t 前进 1、k 只能保持或 +1，终点必须落在第 K-1 行。
    用于把 cross-attention 图（行=关键词 token，列=声学帧）折叠成一个「这段关键词
    是否/在哪里出现」的分数。

    Args:
        matrix: (K, T) 的 numpy 数组或嵌套列表，K 为关键词长度，T 为帧数。

    Returns:
        (max_sum, (start, end))：路径分数和，以及路径覆盖的帧区间。路径整段都待在末行
        （末行注意力压倒性地强，典型是 `<peok>` 当 attention sink）时退化成 `(0, 1)`。

    Note:
        `prev[*][0]` 恒为 None（内层 t 从 1 开始），原实现两处回溯都没判这个边界：只要
        最优路径沿末行一路退回 t=0，就会 `k, t = prev[k][t]` 解包 None 崩掉
        （`TypeError: cannot unpack non-iterable NoneType`）。这里加了 None 守卫，
        `max_sum` 不受影响，只影响退化情形下的 start/end。上游 KCE
        `kce/src/kce/eval/attention/metrics.py:105` 同样有这个缺陷。
    """
    K, T = matrix.shape
    dp = [[0.0] * T for _ in range(K)]
    prev = [[None] * T for _ in range(K)]

    for t in range(T):
        dp[0][t] = matrix[0][t]

    for k in range(1, K):
        for t in range(1, T):
            if dp[k - 1][t - 1] > dp[k][t - 1]:
                dp[k][t] = dp[k - 1][t - 1] + matrix[k][t]
                prev[k][t] = (k - 1, t - 1)
            else:
                dp[k][t] = dp[k][t - 1] + matrix[k][t]
                prev[k][t] = (k, t - 1)

    max_sum = max(dp[K - 1])
    k, t = K - 1, np.argmax(dp[-1])
    while k == K - 1 and prev[k][t] is not None:
        k, t = prev[k][t]
    end = t + 1
    while k > 0 and t > 0 and prev[k][t] is not None:
        k, t = prev[k][t]
    start = t
    return float(max_sum), (start, end)

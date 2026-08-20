# Qwen3.5 MUSA ZeRO-1 原生 AVG 与梯度 Pack 消除

## 1. 结论先行

当前 Qwen3.5-4B 训练在 4 机 × 8 卡、单卡 `bs=4` 的 ZeRO-1 基线上，默认启用 MUSA/MCCL 原生 `AVG` 梯度归约：

```yaml
trainer:
  musa_zero1_native_avg: true
```

它把 DeepSpeed 原来的：

```text
BF16 梯度桶 div_(world_size)
  -> 按 rank 切片并 torch.cat/flatten
    -> AllReduce SUM
```

替换为：

```text
已有的连续 BF16 梯度桶
  -> AllReduce AVG
```

两条路径都计算所有数据并行 rank 的平均梯度。新路径没有减少 AllReduce 的数据量，也没有把通信与 backward 重叠；它删除的是通信前对约 8 GB 梯度桶的两次额外完整遍历，以及 `cat` 所需的临时缓冲区。

同环境 120 个稳态 step 的单变量 A/B 结果：

| 指标 | 原路径 | 原生 AVG | 变化 |
| --- | ---: | ---: | ---: |
| 平均 model time | 602.617 ms | 574.078 ms | -28.540 ms，-4.736% |
| 中位 model time | 599.638 ms | 571.600 ms | -28.038 ms |
| 平均 MFU | 11.620% | 12.197% | +0.577 个百分点 |
| 更快的配对 step | - | 115/120 | 95.8% |
| step 130 loss | 0.224609375 | 0.224609375 | 均为 finite |

## 2. 初学者视角：梯度为什么要取平均

数据并行训练中，每张卡保存相同模型，但读取不同样本。第 $r$ 张卡通过 backward 得到本地梯度 $g_r$。如果数据并行组共有 $N$ 张卡，优化器应使用平均梯度：

$$
g_{\mathrm{avg}} = \frac{1}{N}\sum_{r=0}^{N-1}g_r.
$$

可以把每张卡的梯度理解成一份“局部意见”。AllReduce 会把所有卡的意见合并，并把结果发回每张卡。常见实现有两种：

旧路径先除后求和：

$$
\operatorname{SUM}\left(\frac{g_0}{N},\frac{g_1}{N},\ldots,\frac{g_{N-1}}{N}\right)
= \frac{1}{N}\sum_{r=0}^{N-1}g_r.
$$

新路径让 collective 直接求平均：

$$
\operatorname{AVG}(g_0,g_1,\ldots,g_{N-1})
= \frac{1}{N}\sum_{r=0}^{N-1}g_r.
$$

数学结果相同，但执行方式不同。前一种方法需要每张卡先完整读取并改写自己的大梯度桶；后一种方法把缩放合并到通信 collective 中。

## 3. 梯度桶与 ZeRO-1

### 3.1 梯度桶是什么

模型有很多参数，每个参数都有一块梯度。如果为每个小梯度单独发起通信，会产生大量短 collective 和调度开销。DeepSpeed 先把多个梯度组织进连续的 IPG（Independent Partition Gradient）buffer，再以较大的 bucket 通信。

可以把 bucket 理解成一个已经打包好的大包裹：

```text
parameter 0 grad ┐
parameter 1 grad ├─> contiguous IPG bucket ─> collective
...              │
parameter K grad ┘
```

当前 Trace 中最大 bucket 为：

```text
numel = 3,985,522,112 BF16 elements
bytes = 3,985,522,112 × 2
      = 7,971,044,224 bytes
      ≈ 7.97 GB ≈ 7.42 GiB
```

### 3.2 ZeRO-1 没有被改成 ZeRO-2

ZeRO-1 主要切分 optimizer state。参数和训练所需梯度仍通过数据并行 collective 保持一致。本优化设置 optimizer 实例的 `reduce_scatter=false`，让已有连续 IPG bucket 进入短的 AllReduce reducer；这不等于把训练模式改成 ZeRO-2，也不改变参数、optimizer state 或 checkpoint 的分片格式。

新路径仍对完整连续 bucket 做 AllReduce。这里的收益不是“只传了一个 rank 的分片”，而是避免为当前最终仍执行完整 AllReduce 的路径提前切片并再次拼接。

## 4. 旧路径为什么会出现 `div_` 和 `cat`

DeepSpeed 的默认 `average_tensor` reduce-scatter 风格分支可简化为：

```python
tensor.div_(data_parallel_world_size)
rank_slices = split_and_group_by_destination(tensor)
flat = torch.cat(rank_slices)
all_reduce(flat, op=SUM)
```

其中：

1. `div_` 在本卡先把每个 BF16 梯度除以 world size；
2. rank-slice 逻辑计算每个目标 rank 对应的片段；
3. `flatten`/`torch.cat` 把片段物化为新的连续 tensor；
4. 当前 `use_multi_rank_bucket_allreduce` 路径最终仍执行完整 AllReduce SUM。

对当前大 bucket，这意味着通信开始前有两次额外的全量内存处理：

```text
7.97 GB bucket
  -> div_：完整读写一次
  -> cat：读取切片并写出新的连续 buffer
  -> AllReduce：才开始真正的跨卡通信
```

Trace 中对应的单次关键路径为：

```text
BF16 gradient bucket div_(32)        ~13.03 ms
  -> rank-slice flatten/cat          ~12.31 ms
    -> 7.97 GB AllReduce SUM         ~71.43–75.46 ms
      -> optimizer
        -> parameter AllGather
```

`div_`、`cat` 和大 AllReduce 完全串行，因此前两段约 25 ms 直接增加 step 尾部延迟。`cat` 理论上还需要一块接近 7.97 GB 的目标 buffer；本轮没有记录可归因的峰值显存 A/B，所以这里只说明该临时物化需求被消除，不宣称实测峰值显存降低了多少。

## 5. 新路径如何消除额外遍历

安装器在 `Accelerator.prepare` 创建好实际 DeepSpeed optimizer 后，只修改这个 optimizer 实例：

```python
zero_optimizer.reduce_scatter = False
zero_optimizer.gradient_reduction_w_predivide = native_avg_reducer
```

新的 reducer 接收 DeepSpeed 已有的连续 IPG bucket，并执行：

```python
deepspeed.comm.all_reduce(
    tensor_to_allreduce,
    op=deepspeed.comm.ReduceOp.AVG,
    group=zero_optimizer.dp_process_group,
)
```

完整调用链为：

```text
VLATrainer 初始化
  -> setup_distributed_training / Accelerator.prepare
    -> configure_musa_zero1_native_avg
      -> 找到当前 DeepSpeed ZeRO optimizer
        -> install_musa_zero1_native_avg
          -> reduce_scatter = false
          -> 仅替换该实例的 gradient_reduction_w_predivide
            -> deepspeed.comm.all_reduce(op=AVG)
              -> ProcessGroupMCCL
                -> MCCL AllReduce PreMulSum kernel
```

目标 Trace 实际观察到 kernel 从：

```text
mcclKernel_AllReduce_RING_SIMPLE_Sum___mt_bfloat16
```

变为：

```text
mcclKernel_AllReduce_RING_SIMPLE_PreMulSum___mt_bfloat16
```

`PreMulSum` 表明 MCCL 把缩放合并进 collective，而不是在 collective 前单独发射一个全桶 `div_`。

## 6. 为什么不能只凭公式认定数值等价

实数域中“先除后加”与“先加后除”等价，但 BF16 的有效位有限，每一步运算都可能舍入：

```text
旧路径：round_bf16(g_r / N) -> SUM
新路径：MCCL AVG / PreMulSum 的内部归约与缩放顺序
```

因此两条路径理论上可能产生微小差异，必须做真实 dtype、shape 和拓扑的数值门禁。

当前 world size 为 32：

$$
\frac{1}{32}=2^{-5}.
$$

它是二进制幂，缩放因子可被二进制浮点精确表示，这有助于解释当前 BF16 benchmark 为什么得到逐元素一致结果。但这不是对所有 world size、dtype、MCCL 版本和归约拓扑的永久保证，环境升级后仍需重测。

还有一个必须避免的错误是“双重平均”：如果仍保留原来的 `div_(N)`，然后再执行 `AVG`，结果会变成：

$$
\frac{1}{N^2}\sum_{r=0}^{N-1}g_r,
$$

梯度会错误缩小 $N$ 倍。当前实现通过 `reduce_scatter=false` 绕过原分支中无条件的预除，再在替换后的 reducer 中只做一次 `AVG`。

## 7. 实现边界与 fail-closed

只有以下条件全部满足时才安装：

- MUSA 可用；
- `Accelerator.prepare` 后能找到 DeepSpeed ZeRO optimizer；
- `partition_gradients=false`，即当前仅支持 ZeRO-1；
- `overlap_comm=false`；
- `sequence_parallel_size=1`。

显式开启但不满足边界时会抛出错误，不会静默换成未经验证的路径。实例级安装还具有三个作用：

- 不修改 site-packages 中的 DeepSpeed；
- 不影响同一 Python 进程中的其他 optimizer 实例；
- 可以恢复原 reducer 和原 `reduce_scatter` 值。

执行路径五态证据如下：

| 状态 | 证据 |
| --- | --- |
| imported | `train_starvla.py` 导入配置函数 |
| reachable | `Accelerator.prepare` 后解析出实际 ZeRO optimizer |
| default-on | Qwen3.5 MUSA YAML 中开关为 `true` |
| observed | 启动 warning、Trace 的 `PreMulSum`、大 `div_`/`cat` 消失 |
| fallback | YAML 设为 `false`；恢复原 reducer，不进入 native AVG |

## 8. Trace 证据：删除了什么，没改变什么

基线 Trace：

```text
local_traces/20260820_qwen35_bs4_visionflash_A11_trace_shapes/
  rank_00/worker32016_rank00.1787195378505782728.pt.trace.json.gz
```

目标 Trace：

```text
local_traces/20260820_qwen35_bs4_zero1_native_avg_A11_trace_shapes/
  rank_00/worker32016_rank00.1787196540424892426.pt.trace.json.gz
```

三个 profiler step 的平均结果：

| 指标 | OFF | ON | 差异 |
| --- | ---: | ---: | ---: |
| profiler wall time | 693.817 ms | 641.945 ms | -51.872 ms，-7.48% |
| GPU active union | 517.569 ms | 479.026 ms | -38.544 ms，-7.45% |
| collective GPU 时间 | 142.914 ms | 134.329 ms | -8.585 ms |
| `aten::cat` | 18.415 ms | 4.111 ms | -14.304 ms |
| `aten::div_` | 14.994 ms | 不再是热点 | 大 bucket `div_` 消失 |

关键边界：

- `aten::div_ [[3985522112], []]` 消失；
- 28 个 `141852048` 元素切片对应的大 `flatten_dense_tensors/cat` 消失；
- 大 AllReduce 单体约为 `73.15 ms -> 74.43 ms`，没有变快；
- compute/communication overlap 仍为 `0 ms`。

所以 A11 的原理结论是“删除通信前处理”，不是“减少通信量”“加速网络”或“实现通信重叠”。Profiler 的 `record_shapes` 会放大 launch 间隙，最终收益以无 profiler A/B 为准。

## 9. 数值与性能门禁

### 9.1 Collective microbenchmark

| 拓扑与 shape | pre-divide + SUM | native AVG | 数值结果 |
| --- | ---: | ---: | --- |
| 8 卡，`1,048,576` 元素 | 0.655 ms | 0.625 ms | 逐元素一致，finite |
| 32 卡，`141,852,048` 元素 | 4.750 ms | 4.280 ms | relative L2=0，max abs=0 |
| 32 卡，`3,985,522,112` 元素 | 101.095 ms | 78.386 ms | relative L2=0，max abs=0 |

真实大 bucket benchmark 为 1.290×，并同时验证 32 卡 MCCL `AVG` 无 hang、无 NaN/Inf。

### 9.2 四机训练 A/B

条件固定为：

```text
4 nodes × 8 MUSA devices
per-device batch size = 4
global batch size = 4 × 32 = 128
BF16
text attention = eager
Gated DeltaNet = FLA
vision attention = vision-only FlashAttention
DeepSpeed = ZeRO-1, overlap_comm=false
```

只切换 `trainer.musa_zero1_native_avg`，比较 step 11–130。120 个配对 step 的 language、vision 和 action token shape 完全一致：

| 指标 | OFF | ON | 结论 |
| --- | ---: | ---: | --- |
| 平均 model time | 602.617 ms | 574.078 ms | 改善 4.736% |
| 标准差 | 11.019 ms | 10.611 ms | 波动未恶化 |
| 配对 delta 标准差/SEM | - | 15.628/1.427 ms | 收益约为 20× SEM |
| finite loss | 是 | 是 | 通过 |
| step 11 loss | 0.58203125 | 0.6015625 | 不要求逐 step 相同 |
| step 130 loss | 0.224609375 | 0.224609375 | 下降趋势一致 |

不同训练轮次即使 seed 和 shape 固定，也不应要求每一步 loss 逐 bit 相同；门禁关注非有限值、下降趋势和短训终点是否合理。当前 collective microbenchmark 的输出是精确一致的，训练 loss 也通过 finite 与趋势检查。

## 10. 验证命令

实例安装、AVG 调用、dtype copy、回退和边界测试：

```bash
python -m pytest -q tests/test_musa_zero1_native_avg.py
```

单机 8 卡 collective benchmark：

```bash
torchrun --standalone --nproc_per_node=8 \
  tests/benchmark_musa_allreduce_avg.py \
  --numel 1048576 \
  --warmup 2 \
  --iters 5
```

真实大 bucket 应在与训练一致的 32 卡 torchrun 拓扑下执行。以下命令需要在 4 个节点分别运行，并为每个节点设置不同的 `NODE_RANK`：

```bash
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  tests/benchmark_musa_allreduce_avg.py \
  --numel 3985522112 \
  --warmup 2 \
  --iters 5
```

最后一次组合回归结果为 `13 passed`；同时通过 Ruff、Shell syntax 和 `git diff --check`。

## 11. 回退与升级后复验

立即回退：

```yaml
trainer:
  musa_zero1_native_avg: false
```

以下变化后必须重新执行 collective microbenchmark、Trace 与短训 A/B：

- torch、torch_musa、MCCL 或 DeepSpeed 版本变化；
- world size、数据并行组或网络拓扑变化；
- 梯度通信 dtype 变化；
- ZeRO stage、`reduce_scatter`、`overlap_comm` 或 sequence parallel 配置变化；
- 梯度 bucket 大小或模型参数规模显著变化。

若 Trace 中重新出现大 bucket `div_`/`flatten_dense_tensors`，或没有观察到 `AVG` 对应的 MCCL 路径，应认为优化未实际生效，而不是仅凭 YAML 值继续做性能比较。

## 12. 代码索引

| 文件 | 作用 |
| --- | --- |
| `starVLA/training/musa_zero1_native_avg.py` | 实例级安装、原生 AVG reducer、边界检查和回退 |
| `starVLA/training/train_starvla.py` | distributed prepare 后安装优化 |
| `examples/Robotwin/train_files/starvla_cotrain_robotwin_qwen35_abs.yaml` | 默认开关与唯一配置源 |
| `examples/Robotwin/train_files/run_robotwin_train_qwen3_5_musa.sh` | 启动前解析并打印 YAML 策略 |
| `tests/test_musa_zero1_native_avg.py` | reducer 契约和 fail-closed 单测 |
| `tests/benchmark_musa_allreduce_avg.py` | 真实 MUSA/MCCL BF16 collective 数值与性能门禁 |

## 13. 采用决定

该优化已完成从局部算子到端到端的证据闭环：

- 原理上只消除冗余的全桶缩放与 rank-slice pack；
- 真实 32 卡大 bucket 数值一致且无 hang；
- observed-path Trace 证明 `div_`/大 `cat` 消失并进入 MCCL `PreMulSum`；
- 四机 120-step 单变量 A/B 改善 4.736%，显著高于噪声；
- loss finite 且下降趋势正常；
- YAML 可一键回退，未修改 DeepSpeed 全局状态或 checkpoint 格式。

因此当前 Qwen3.5 MUSA bs=4 基线默认启用 ZeRO-1 native AVG。通信 overlap 仍是独立候选，不能把 A11 的结果视为 overlap 已完成。

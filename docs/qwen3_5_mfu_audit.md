# Qwen3.5-4B RoboTwin 任务的 MFU 计算与 Trace 审计

本文对应 StarVLA 的 Qwen3.5-4B RoboTwin 训练任务，公式版本为
`qwen3_5_v2`。审计配置为 4 节点 × 8 张 MUSA X10000、单卡 batch size 1、
DeepSpeed ZeRO-1、BF16、`sdpa/math`、TF32 关闭、Qwen3.5 FLA fast path 开启、
原生 AdamW。配置中的单卡 BF16 峰值假设为 460 TFLOPS。

## 1. 先用一句话理解 MFU

MFU（Model FLOPs Utilization）回答的是：**这一秒内完成的“有用模型计算”，占一张卡
理论上最多能完成的 BF16 计算的百分之几。**

可以把设备理解成一台理论产能为 460 TFLOPS 的机器：

1. 先估算一个训练 step 需要完成多少“有效工作量”；
2. 再除以这个 step 实际花费的秒数，得到实际工作速率；
3. 最后除以机器的理论峰值。

公式是：

```text
实际吞吐（TFLOPS/卡） = 单卡每 step 的有效 TFLOPs / model_time（秒）
MFU（%）              = 实际吞吐 / 单卡理论峰值 × 100
```

TFLOP/step 是“工作量”，TFLOPS 是“每秒完成的工作量”，不要混淆两者。

## 2. 本任务如何估算一个 step 的有效 FLOPs

静态模型信息由 `train_starvla.py::_build_qwen35_flop_config` 从实际 checkpoint
读取，动态 token 数由 `QwenOFT.forward` 从当前单卡 micro-batch 的真实 tensor shape
读取。最终计算在 `starVLA/training/mfu.py` 中完成。

### 2.1 为什么训练矩阵计算通常是 `6 × 参数量 × token 数`

一次矩阵乘法的前向约为 `2 × 参数量 × token 数` FLOPs。训练还要计算输入梯度和
权重梯度，两个反向矩阵乘法再贡献约 4 倍，因此合计约为 6 倍。这里的 multiply-add
按两个 FLOP 计数。

本任务分别计算：

- 文本层权重：`6 × 文本层参数量 × 语言 token 数`；
- 视觉 block、patch embedding 和 merger 的权重计算；
- action head：`6 × action 参数量 × action token 数`；
- full attention 的 `QKᵀ` 和 `AV`，含反向；
- Qwen3.5 的 Gated DeltaNet 无参数状态更新，不能只靠参数量覆盖，因此单独增加
  `21 × 层数 × token 数 × value_heads × key_dim × value_dim`；
- LM head 只计算最后一个位置，且 logits 不参与 action loss，所以只计前向
  `2 × lm_head 参数量 × logits token 数`，不计 LM head 反向。

视觉 attention 当前假设同一个 micro-batch 中的图片 patch 网格相同。RoboTwin
当前固定 resize，每个样本 3 张图片，因此这个假设在当前任务中成立。

### 2.2 一个可手算、可单测的真实实例

测试 `tests/test_mfu.py::test_robotwin_qwen35_reference_shape` 使用从当前
Qwen3.5-4B checkpoint 读取的参数量，以及一个真实形状：

```text
单卡 batch size                = 1
padding 后语言长度             = 291
原始 vision patch token        = 768（3 张图片）
spatial merge 后 vision token  = 768 / 2² = 192
action token                   = 50
LM head logits token           = 1
```

得到的单卡有效工作量分解如下：

| 部分 | TFLOP/step | 占比 |
|---|---:|---:|
| 文本层矩阵计算 | 6.233306490 | 79.363% |
| full attention | 0.033297924 | 0.424% |
| Gated DeltaNet core | 0.076894175 | 0.979% |
| 视觉 block 矩阵计算 | 1.393041605 | 17.736% |
| 视觉 attention | 0.057982058 | 0.738% |
| 视觉 patch embedding | 0.007252476 | 0.092% |
| 视觉 merger | 0.031416975 | 0.400% |
| action head | 0.019697668 | 0.251% |
| LM head 最后一个位置的前向 | 0.001271398 | 0.016% |
| **总计** | **7.854160770** | **100%** |

假设这个 step 的 `model_time = 0.5 s`，则：

```text
实际吞吐 = 7.854160770 / 0.5
         = 15.708321540 TFLOPS/卡

MFU      = 15.708321540 / 460 × 100
         = 3.414852509%
```

这个结果由 `test_reference_shape_mfu_hand_calculation` 直接验证。

## 3. 为什么 32 卡不需要再乘或再除一次

当前日志中的 FLOPs、吞吐、峰值和 MFU 全部是**单卡口径**。全局 batch size 是 32，
但每个 data-parallel rank 的本地 batch size 是 1，单卡完成约 7.85 TFLOP 的有效工作。

如果改用集群口径，分子和分母必须同时乘 32，结果会抵消：

```text
(7.854 × 32) / 0.5 / (460 × 32)
= 7.854 / 0.5 / 460
```

只给分子或只给分母乘 world size，都会产生 32 倍错误。当前实现没有这个问题。

## 4. `model_time` 是否真的覆盖了设备执行

训练循环在 `_train_step` 前后使用 `time.perf_counter()`。计时范围包括 forward、
backward、梯度裁剪、optimizer 和 scheduler，不包括 dataloader；数据时间单独记为
`data_time`。

MUSA kernel 是异步提交的，但 `_train_step` 返回前执行 `action_loss.item()`，会同步
此前同一执行流上的设备工作，因此 `model_time` 不只是 CPU 提交 kernel 的时间。
这使当前配置下的计时口径成立。

滚动 MFU 不是逐 step 百分比的算术平均，而是：

```text
sum(窗口内有效 TFLOP) / sum(窗口内 model_time) / 峰值
```

默认跳过前 10 个 warmup step，再使用最多 20 个 step 的窗口。本次 8-step profiler
任务故意很短，因此没有滚动 MFU；正常性能基线应运行足够多的非 profiler step。

## 5. 2026-08-17 Trace 实例

运行 ID：`codex_qwen35_trace_20260817_115560`。

Profiler 仅在全局 rank 0 启用，调度为 `wait=2, warmup=1, active=3, repeat=1`，记录
CPU、MUSA、tensor shape、显存和调用栈。4 节点任务完成 8/8 steps，无
OOM、Traceback 或进程异常。

active 的 step 4–6 聚合结果为：

```text
sum(有效 TFLOP) / sum(model_time) = 11.137368 TFLOPS/卡
profiled MFU                       = 2.421167%
平均 model_time                   = 0.701930 s
```

这个 2.42% **不能当作正常训练基线**，因为 shape、memory 和 stack profiler 会显著
增加开销。同一配置此前无 profiler 的稳定区间约为 0.498 s/step、3.41% MFU。

Trace 共包含 1,277,772 个事件和 34,617 个 device kernel，折合每个 active step
约 11,539 次 kernel 启动。方向性结论如下：

- `starvla.backward` 的 GPU 标注合计 1,417.557 ms，约 472.5 ms/step；
- `starvla.forward` 合计 504.221 ms，约 168.1 ms/step；
- MCCL all-reduce 和 all-gather kernel 合计约 398.5 ms，约 132.8 ms/step；
- 大量 identity、cast、copy、逐元素 add/mul 小 kernel 表明启动和访存碎片仍明显；
- Trace 中出现 `chunk_gated_delta_rule_*` 和 `prepare_wy_repr_*`，说明当前 FLA fast
  path 的专用 Gated Delta kernel 确实生效。

这些事件可能存在嵌套或跨 stream 重叠，不能把各项耗时简单相加后当作百分比。
但它们足以把下一轮优化重点放在：backward 内的通信重叠/分桶，以及高频小 kernel
的定向融合。当前日志还明确提示 SDPA flash attention 被 runtime disabled；不过在本任务
约 281–295 的短序列下，attention 的有效 FLOPs 占比较小，应先用 A/B Trace 验证收益再
投入移植工作。

本地 Trace：

```text
C:\works\jd_2026\vla_project\old_starvla\local_traces\
  codex_qwen35_trace_20260817_115560\rank_00\
    profile_config.json
    worker32016_rank00.1786952201143472798.pt.trace.json.gz
```

Trace SHA-256：

```text
2bbc226190db218ae969f88704f10108326ce4db5f49377f56a6573e72235095
```

## 6. 审计结论与边界

结论：`qwen3_5_v2` 对当前 Qwen3.5-4B RoboTwin 配置的计算口径自洽，真实动态 shape、
混合 full/linear attention、vision、action 和只保留一个位置的 LM head 均已覆盖；
单卡/集群口径及计时同步也正确。本轮把最终吞吐和 MFU 换算抽成
`calculate_per_device_mfu`，使手算值和非法时间/峰值能被单元测试直接验证。

仍需牢记以下边界：

- 460 TFLOPS 是配置假设，不是运行时自动查询值；换卡或改变精度后必须同步修改；
- 分子只统计有用模型 FLOPs，不统计 optimizer、通信、逐元素操作和 gradient
  checkpointing 的重计算，但这些工作花费的时间仍在分母中；因此这是 MFU，不是统计
  实际执行 FLOPs 的 HFU；
- `6 × 参数 × token` 假设对应训练中的矩阵参数；若以后冻结大块模型，需要重新审查
  权重梯度是否还应按完整 6 倍计算；
- 当前视觉 attention 的均匀图片网格假设只对固定 resize 数据成立；
- MFU 适合比较固定模型、数据 shape、精度和峰值口径下的 A/B，不应跨口径直接排名。

验证命令：

```bash
python -m unittest discover -s tests -p 'test_mfu.py' -v
```

当前共 5 个测试，覆盖真实参考形状、batch 线性缩放、手算 MFU、非法时间/峰值和非法
vision shape。

# StarVLA Qwen3.5-4B MUSA 适配说明

## 1. 文档目标

本文档说明如何在 StarVLA 中使用 `Qwen3.5-4B + QwenOFT + RoboTwin + DeepSpeed + MUSA`，以及每个适配点背后的原因。

阅读本文档不要求了解 Transformer、视觉语言模型或机器人算法。对 AI Infra 工程师来说，可以把这条链路理解为：

```text
图像 + 文字指令
        │
        ▼
Qwen3.5 Processor：把图像和文字变成张量
        │
        ▼
Qwen3.5-4B：让每个 token 拥有“理解图像和指令后的特征”
        │
        ▼
取出 50 个动作占位 token 对应的特征
        │
        ▼
QwenOFT MLP Action Head：每个特征预测一帧机器人动作
        │
        ▼
预测动作与数据集真值计算 L1 loss
```

当 loss 能够稳定下降时，表示模型预测的动作在逐渐接近数据集中的真实动作。

## 2. 本分支的适用边界

- 开发分支：`gl/musa_dev-qwen3.5`。
- 模型：`Qwen/Qwen3.5-4B`，本地训练时使用已下载的权重目录。
- VLA 框架：`QwenOFT`，不是 `QwenGR00T`。
- 数据集：RoboTwin，默认 `robotwin_all_50`。
- 分布式：Accelerate 调用 DeepSpeed ZeRO-1，单机 8 卡或多机每机 8 卡。
- 设备后端：原生 `torch_musa`/MUSA。本分支没有显式导入 `torchada`，因此不能假设所有 `cuda` API 都会自动映射到 MUSA。

上游 [PR #172](https://github.com/starVLA/starVLA/pull/172) 主要验证了 Qwen3.5 与 QwenGR00T；[Issue #291](https://github.com/starVLA/starVLA/issues/291) 说明 QwenOFT 路径的动作占位 token 需要单独修正。所以“上游支持 Qwen3.5”不等于“QwenOFT 在 MUSA 上无需适配”。

## 3. 文件与职责

| 文件 | 作用 |
| --- | --- |
| `requirements.txt` | 固定 `transformers==5.2.0` |
| `pyproject.toml` | 包安装时的 Transformers 依赖和可选 FLA 0.4.2 依赖 |
| `starVLA/model/modules/vlm/__init__.py` | 根据显式 `model_type` 或权重路径选择 Qwen3.5 wrapper |
| `starVLA/model/modules/vlm/QWen3_5.py` | 加载模型、Processor、autocast 和 SDPA 后端 |
| `starVLA/model/modules/vlm/qwen35_musa.py` | 控制 Qwen3.5 Gated DeltaNet 的 MUSA FLA/参考路径 |
| `starVLA/model/framework/QwenOFT.py` | 构造动作 token、抽取 hidden state、计算动作 loss |
| `starVLA/training/mfu.py` | Qwen3.5 每卡每 step FLOPs 估算 |
| `starVLA/training/train_starvla.py` | DeepSpeed 训练、fused optimizer、TF32、Profiler 和 MFU 打印 |
| `examples/Robotwin/train_files/starvla_cotrain_robotwin_qwen35_abs.yaml` | Qwen3.5 QwenOFT 训练参数 |
| `examples/Robotwin/train_files/run_robotwin_train_qwen3_5_musa.sh` | 单节点入口，也被每个多机 launcher 调用 |
| `cluster/dist_run_qwen3_5.sh` | 根据 hostfile 在多个节点上拉起入口脚本 |

## 4. Transformers 与权重版本

### 4.1 为什么固定 5.2.0

StarVLA 通过 `Qwen3_5ForConditionalGeneration` 加载 Qwen3.5，因此运行时 Transformers 必须提供这个类。本分支把环境基线固定为：

```text
transformers==5.2.0
```

权重 `config.json` 中可能仍然看到类似 `"transformers_version": "4.57.0.dev0"` 的内容。这是权重导出时记录的元数据，不代表当前 Python 进程实际导入了 4.57.0。实际版本应这样检查：

```bash
python -c 'import sys, transformers; print(sys.executable); print(transformers.__version__)'
```

多机训练时，每个节点的 Python 路径和版本都必须一致。

### 4.2 权重路径不应决定模型类型

旧选择逻辑只检查路径字符串是否包含 `Qwen3.5`。如果权重放在 `/data/models/latest`，代码就无法判断它是 Qwen3.5。

当前配置显式增加：

```yaml
framework:
  qwenvl:
    model_type: qwen3_5
    base_vlm: /actual/path/to/Qwen3.5-4B
```

`model_type` 决定使用哪个 wrapper，`base_vlm` 只负责告诉 Transformers 到哪里读权重。

## 5. QwenOFT 动作 token 适配

### 5.1 动作 token 是什么

QwenOFT 需要一次预测 50 帧动作，所以在 prompt 中连续插入 50 个占位 token。模型计算完成后，代码取出这 50 个位置的 hidden state，每个 hidden state 预测一帧动作。

这意味着占位符必须满足两个条件：

1. 单独编码时恰好是 1 个 token。
2. 连续重复 50 次时，仍然是 50 个相同 token，不能被 BPE 合并为其他 token。

原占位符 `🔍` 在 Qwen3.5 tokenizer 中会被拆分，会导致代码取错 hidden state，即使训练不立即报错，loss 也可能异常。

当前配置使用 Qwen3.5 tokenizer 中已保留的 `<|fim_pad|>`：

```yaml
action_token: "<|fim_pad|>"
```

模型初始化时会同时校验单 token 和连续重复结果。不满足条件时直接报错，防止带着错误的动作位置进入长跑。

### 5.2 动作标签长度

```text
chunk_len = past_action_window_size + 1 + future_action_window_size
```

当前配置是 `0 + 1 + 49 = 50`。预测和标签都必须取 `chunk_len` 帧。如果只按 `future_action_window_size + 1` 切标签，当以后将 `past_action_window_size` 设为非 0 时，预测与标签的时间维就不一致。

### 5.3 Action Head 的 dtype

动作头在裸模型初始化后通常是 FP32；DeepSpeed BF16 初始化后可能变为 BF16。因此不应假设它永远与 Qwen 输出 dtype 一致。

当前实现在 Action Head 前读取其实际参数 dtype，并将 action query 转为相同 dtype。如果外层 DeepSpeed/autocast 正在工作，Linear/GEMM 仍然可按 autocast 策略选择 BF16；这里的目的是防止裸模型或回退路径出现 BF16 输入与 FP32 权重直接 MatMul 的类型错误。

`torch.autocast(..., dtype=torch.float32)` 不能用来“强制 FP32”：FP32 通常不是加速器 autocast 支持的目标 dtype，容易得到 warning 或直接禁用 autocast。

## 6. Qwen3.5 前向适配

### 6.1 为什么关闭 KV cache

KV cache 是自回归文本生成时的加速缓存。QwenOFT 训练不做逐 token 文本生成，只读 hidden state，所以训练前向固定：

```python
use_cache = False
```

这可以避免不需要的 cache 分配，也与 gradient checkpointing 的训练语义一致。

### 6.2 为什么设置 `logits_to_keep=1`

Qwen3.5 原本会将每个序列位置投影到约 24.8 万词表，产生 `[batch, sequence, vocabulary]` 的大 logits 张量。QwenOFT 的动作 loss 并不使用这些 logits。

当前实现保留最后 1 个位置的 logits，避免生成整个大张量：

```python
logits_to_keep = 1
```

这不会切断 hidden state 到 Action Head 的梯度；只是减少不参与 action loss 的 LM Head 计算。

## 7. MUSA Attention 路径

Qwen3.5-4B 是混合架构：32 层文本层中，8 层是普通 full attention，24 层是 Gated DeltaNet 线性注意力。这两类层使用不同的后端。

### 7.1 Full attention：默认 SDPA math

Qwen3.5-4B full attention 的 `head_dim=256`。当前目标 MUSA 环境中，SDPA 自动选择 flash 路径时存在 RoPE/注意力执行异常风险，所以正确性基线为：

```yaml
attn_implementation: sdpa
sdpa_backend: math
```

wrapper 在每次 Qwen3.5 `forward`/`generate` 外使用 `sdpa_kernel(SDPBackend.MATH)`，限制本次调用，不修改 Transformers 安装目录。

安全回退是 `eager`。之后如果新版 MUSA FlashAttention 已支持当前 shape，再将 `sdpa_backend` 切换为 `auto` 做单变量 A/B，不要在长跑中盲目切换。

### 7.2 Gated DeltaNet：参考路径与 FLA

`STARVLA_QWEN35_FLA_FASTPATH` 有三种语义：

| 值 | 行为 | 用途 |
| --- | --- | --- |
| `0` | 在 MUSA 上强制 Transformers 参考 Gated DeltaNet、参考 RMSNorm 和 PyTorch causal conv | 正确性基线、问题隔离 |
| `1` | 要求 FLA 0.4.2，替换训练序列路径；缺依赖时报错 | 经数值 A/B 后的性能路径 |
| `auto` | 保留 Transformers 导入时的自动选择 | 库能力探索，不作为可复现基线 |

修正前，开关为 `0` 时只是“不执行 StarVLA 补丁”。如果 Transformers 因为环境中存在 FLA 而已经自动绑定快路径，实际仍可能运行 FLA。当前实现在 MUSA 上会显式重绑，使 `0/1` A/B 的语义可信。

FLA 0.4.2 是当前环境适配基线，不代表它在所有 MUSA/Triton 组合上都已验证。它必须先完成 forward、backward、梯度、短训 loss、显存和长跑验证，再考虑默认打开。

### 7.3 TF32 与 RoPE

本分支不保留 RoPE 算子替换或数值 workaround。`STARVLA_ALLOW_TF32` 只控制 TF32 策略：

- `auto`：保留 PyTorch/torch_musa 默认值。
- `0`：明确关闭 TF32，适合做数值隔离实验。
- `1`：明确打开 TF32。

该开关会同时设置 MUSA 可能继承的 CUDA-compatible matmul flag、muDNN flag 和 PyTorch float32 matmul precision，并在启动时打印结果。

## 8. DeepSpeed 与 Optimizer

当前入口使用：

```text
starVLA/config/deepseeds/deepspeed_zero1_musa.yaml
  └── starVLA/config/deepseeds/ds_config_zero1_musa.yaml
```

DeepSpeed 配置为 BF16 + ZeRO-1。ZeRO-1 主要切分 optimizer state，参数和梯度并不像 ZeRO-3 那样全部切分，优点是路径简单、适合先做正确性和性能基线。

Optimizer 开关：

```text
# 默认值：0；显式设置为 1 才启用 FusedAdamW
STARVLA_ENABLE_FUSED_OPTIMIZER=1
```

当开关为 1 且 MUSA 可用时，代码尝试 `torch_musa.optim.FusedAdamW`；导入失败时记录 warning 并回退到 `torch.optim.AdamW`。在当前 MTT S5000、torch 2.7.1.post1、Transformers 5.2.0 的 4 节点 Qwen3.5 配置中，固定 `sdpa/math`、TF32=0、FLA=1 做 6 step A/B，native AdamW 的稳态 `model_time` 为 0.498 s，FusedAdamW 为 0.635 s；因此当前入口默认关闭 fused。升级 torch_musa、驱动或改变 shape/拓扑后必须重新 A/B，不能仅凭算子名称判断端到端收益。

## 9. MFU 估算与“为什么很低”

### 9.1 MFU 是什么

MFU（Model FLOPs Utilization）用来估计“理论上每秒能做的计算”中，多少真正转化成了模型训练所需的有效计算：

```text
估算每卡实现 TFLOPS = 估算每卡每 step TFLOPs / step 时间
MFU = 估算每卡实现 TFLOPS / 单卡 BF16 稠密理论峰值
```

分子和分母必须使用同一口径：本实现两者都是“每卡”，所以不乘 world size。

### 9.2 Qwen3.5 v2 公式包含什么

当前公式版本为 `qwen3_5_v2`，主要包含：

1. 参数 GEMM：约 `6 × 参数量 × token 数`，对应前向、激活梯度和权重梯度。
2. 8 层 full attention 的 `QKᵀ` 和 `attention·V` 矩阵计算。
3. 24 层 Gated DeltaNet 的无参数状态读、状态更新和输出读取。
4. 视觉 patch embed、24 层视觉 block、视觉 attention 和 merger。
5. QwenOFT Action Head。
6. `logits_to_keep=1` 对应的单位置 LM Head 前向。该 logits 不参与 action loss，所以不计 LM Head backward。

它不把 optimizer、通信、elementwise 算子或 gradient checkpointing 带来的额外重算计入 MFU 分子。forward、backward、通信、梯度裁剪和 optimizer 的开销都包含在 `model_time` 分母中；数据加载由外层单独计为 `data_time`，不包含在 `model_time` 中。这是常见的“有效模型 FLOPs”口径；如果把重算也放进分子，更接近 HFU 而不是本文的 MFU。

### 9.3 当前参考 shape

单卡 batch=1、padding 后文本序列 291、3 张图共 768 个原始视觉 patch token、50 个动作 token 时：

```text
qwen3_5_v1 估算：约 7.777 TFLOPs/卡/step
qwen3_5_v2 估算：约 7.854 TFLOPs/卡/step
其中 Gated DeltaNet 核心补计：约 0.077 TFLOPs/卡/step
```

所以旧公式确实偏低，但漏算量不足总 FLOPs 的 1%，不能解释数倍的 MFU 偏低。

例如每 step 需要 2 秒，单卡理论峰值填 460 TFLOPS：

```text
估算实现算力 = 7.854 / 2 = 3.927 TFLOPS
MFU = 3.927 / 460 × 100% ≈ 0.85%
```

这个数学结果虽然很低，但不一定是打印 bug。

历史内网 8 卡 FLA 长跑的稳态中位数约为 `0.843 秒/step`。用 v2 公式复算：

```text
估算实现算力 = 7.854 / 0.843 ≈ 9.32 TFLOPS/卡
MFU = 9.32 / 460 × 100% ≈ 2.03%
```

这与旧日志打印的约 `2.01%` 基本一致，说明打印链路没有数量级错误。该任务同时出现约 99% 的 device utilization 也不矛盾：device utilization 只表示设备大部分采样时间处于忙碌状态，不表示矩阵核心达到了理论峰值。短序列、小 batch、许多小算子、重算及通信都可能让设备“很忙”，但有效 GEMM 吞吐仍然很低。

### 9.4 MFU 偏低的常见原因

建议按以下顺序检查：

1. `gpu_peak_tflops` 是否填了单卡 BF16 稠密峰值。不能填稀疏峰值，也不能填 8 卡或整机总峰值。
2. 每卡 batch=1 且序列只有约 291 token，大 GEMM 很难充分占满设备。
3. SDPA math 是正确性基线，通常比已验证的 flash attention 慢。
4. `STARVLA_QWEN35_FLA_FASTPATH=0` 使用参考 Gated DeltaNet，它用大量 PyTorch 操作，性能可能显著低于 FLA。
5. gradient checkpointing 会在 backward 重做一部分 forward，额外时间不进入本 MFU 分子。
6. ZeRO、MCCL 通信、gradient clipping、optimizer 都在 step 时间内，但不计为模型有效 FLOPs。
7. Python/CPU 发射空隙和设备同步会拉长 `model_time`；数据解码只会拉长单独打印的 `data_time`，不会直接降低这里的 MFU。

### 9.5 打印口径

训练会同时打印：

- `estimated_tflops_per_device_step`：当前 shape 的每卡每 step 估算 TFLOPs。
- `estimated_text_tflops_per_device_step`：文本骨干与单位置 LM Head 的估算 TFLOPs。
- `estimated_vision_tflops_per_device_step`：视觉塔的估算 TFLOPs。
- `estimated_action_tflops_per_device_step`：动作头的估算 TFLOPs；三部分相加应等于总数。
- `achieved_tflops_per_device`：当前 step 的估算实现 TFLOPS。
- `mfu_percent`：当前 step 的即时 MFU，波动可能较大。
- `achieved_tflops_per_device_rolling`：预热后窗口内的 `累计 FLOPs / 累计时间`。
- `mfu_percent_rolling`：建议用于性能对比的稳态 MFU。
- `mfu_formula`：公式版本，防止将 v1 和 v2 直接比较。

默认跳过 10 个冷启动 step，使用最近 20 个 step 计算 rolling MFU。训练 step 返回前读取 `action_loss.item()`，该操作会等待当前设备上与 loss 相关的排队工作，所以外层 `perf_counter()` 不是纯 Python 发射时间。如果要对异步 stream 和 MCCL 做精确分解，仍需要 Trace，不能只看 MFU。

## 10. 启动方式

### 10.1 启动前检查

```bash
cd /path/to/starVLA

python -c 'import torch, torch_musa, transformers, accelerate, deepspeed; \
print("torch", torch.__version__); \
print("transformers", transformers.__version__); \
print("musa", torch.musa.is_available(), torch.musa.device_count())'

test -f /actual/path/to/Qwen3.5-4B/config.json
test -d /actual/path/to/RoboTwin
```

模型目录至少应包含 `config.json`、tokenizer/processor 文件和全部权重分片。数据目录需要是 StarVLA RoboTwin dataloader 能识别的已解压、已预处理结构，不能只有下载压缩包。

### 10.2 单机 8 卡

```bash
BASE_VLM=/actual/path/to/Qwen3.5-4B \
DATA_ROOT_DIR=/actual/path/to/RoboTwin \
RUN_ID=qwen35_robotwin_smoke \
MAX_TRAIN_STEPS=10 \
SAVE_INTERVAL=1000 \
EVAL_INTERVAL=1000 \
STARVLA_QWEN35_FLA_FASTPATH=0 \
STARVLA_ALLOW_TF32=auto \
bash examples/Robotwin/train_files/run_robotwin_train_qwen3_5_musa.sh
```

先用 10 step 验证正确性，再增大 `MAX_TRAIN_STEPS`。如果希望验证 FLA，只改 `STARVLA_QWEN35_FLA_FASTPATH=1`，其他参数保持不变。

### 10.3 多机

hostfile 每行一个可通过 SSH key 访问的主机名或 IP：

```text
worker0
worker1
worker2
worker3
```

启动：

```bash
BASE_VLM=/home/jd/gl_dev/models/Qwen3.5-4B \
DATA_ROOT_DIR=/actual/shared/path/RoboTwin \
WORKDIR=/home/jd/gl_dev/starVLA \
bash cluster/dist_run_qwen3_5.sh cluster/hostfile
```

launcher 会先检查每个节点的 SSH、工作目录、入口脚本、模型和数据路径，再拉起每台机器的 Accelerate launcher。所有节点必须看到同一份代码、权重、数据和输出目录。

## 11. 验证顺序

不要从“能创建进程”直接跳到长跑。推荐顺序如下：

1. 静态检查：Python compile、Shell syntax、`git diff --check`。
2. 环境检查：每个 rank 的 Python、Transformers、torch_musa、DeepSpeed、FLA 版本一致。
3. tokenizer 检查：`<|fim_pad|>` 单独和重复 50 次的 token id 符合预期。
4. Processor 检查：一个真实 RoboTwin batch 能产生 `input_ids`、`pixel_values`、`image_grid_thw`。
5. 单卡 1 step：forward、backward、optimizer 都成功，loss 有限且梯度非空。
6. 单机 8 卡 10 step：所有 rank 正常，loss 无 NaN/Inf，能打印即时 MFU。
7. 单机稳态 A/B：固定 shape、seed 和数据，比较 reference/FLA、SDPA math/eager、fused/unfused optimizer。
8. 多机短测：检查 MCCL、rank 日志、step 一致性和 rolling MFU。
9. 长跑：同时监控 loss 趋势、梯度、step time、峰值显存、MFU 和 checkpoint 可恢复性。

性能验收至少要跳过冷启动，统计多个稳态 step 的平均值和波动范围。不能用首 step 或单 step 宣称性能收益。

## 12. 已知限制

- 本地 Windows Python 环境不包含 torch/torch_musa，所以本地只能完成纯 Python MFU 单测和静态 compile；MUSA 数值、性能和分布式结论必须在目标 Pod 中验证。
- FLA 0.4.2 仍是实验性快路径，默认关闭。
- SDPA math 是已知正确性基线，不代表最佳性能。
- MFU 是基于模型结构和当前 batch shape 的估算，不能替代 MUSA Trace 或算子 profiler。
- 如果更改 Transformers、torch_musa、Triton、FLA、图像分辨率、batch、序列长度或分布式拓扑，需要重新执行数值、性能、显存和稳定性验收。

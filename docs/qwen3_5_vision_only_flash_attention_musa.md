# Qwen3.5 Vision-only FlashAttention：MUSA 原理、实现与验证

## 1. 结论先行

StarVLA 的 Qwen3.5-4B 训练默认保持模型级 attention 为 eager，但只把视觉编码器的 `Qwen3_5VisionAttention` 切到 MUSA packed varlen FlashAttention：

```yaml
framework:
  qwenvl:
    attn_implementation: eager
    sdpa_backend: auto
    musa_vision_flash_attention: true
```

最终执行边界是：

```text
文本 full attention，head_dim=256
  -> Transformers eager
  -> 不进入 SDPA flash/math
  -> 不进入 Mate/TileLang

视觉 attention，head_dim=64
  -> musa_flash_varlen
  -> MUSA flash-attn extension
  -> Mate 调用数为 0
```

当前 bs=4 真实训练 shape 的结果：

| 指标 | eager vision | vision-only Flash | 结果 |
| --- | ---: | ---: | ---: |
| 真实 `12×256` forward 中位数 | 2.776 ms | 0.710 ms | 3.91× |
| 真实 `12×256` forward+backward 中位数 | 7.078 ms | 1.735 ms | 4.08× |
| 四机 step 11–130 平均 model time | 750.123 ms | 603.548 ms | 改善 19.54% |
| 四机平均 MFU | 9.335% | 11.602% | +2.267 个百分点 |
| 四机 loss | finite | finite | 通过 |

MFU 与算子效率使用 BF16/FP16 `460 TFLOPS/卡` 作为参考分母。S5000 的 `1000 TFLOPS` 是 FP8 规格，不能用于本训练的 BF16/FP16 计算。

## 2. 为什么只能先融合视觉 attention

Qwen3.5-4B 包含三类不同的数据流：

1. 文本 full attention：8 层，GQA，`head_dim=256`；
2. 文本 Gated DeltaNet：24 层，当前使用 FLA；
3. 视觉 self-attention：24 个 block，MHA，`head_dim=64`。

此前验证过的风险集中在文本 full attention：

- SDPA flash/math 的 mask 路径出现过非有限值；
- 直接 FlashAttention 的 `head_dim=256` backward 会进入 Mate/TileLang；
- Mate/TileLang 路径在实际多卡训练中发生过 hang。

视觉 attention 的约束不同：

- `head_dim=64`；
- Q/K/V head 数相同，不需要 GQA repeat；
- 每张图内部是非 causal attention；
- 图像之间用 `cu_seqlens` 隔离，不需要文本的 padding/causal 加法 mask；
- 当前 flash-attn 包对该 shape 走 MUSA extension，而不是 Mate。

因此“只切视觉、不动文本”不是临时拼接，而是按模型子图、shape 和已验证 backend 能力划出的安全边界。

## 3. eager 视觉 attention 的计算过程

### 3.1 真实 shape

当前单卡 batch 为 4，每个 step 的训练日志记录：

```text
vision_patch_tokens_per_device = 3072
```

每个视觉序列有 256 个 token，因此每卡为 12 段 packed sequence：

```text
3072 / 256 = 12
```

视觉配置为：

```text
hidden_size = 1024
num_heads   = 16
head_dim    = 1024 / 16 = 64
```

QKV projection 后的逻辑 shape 为：

```text
hidden_states: [3072, 1024]
q/k/v:         [3072, 16, 64]
```

Transformers attention 接口使用 `[batch, heads, sequence, head_dim]`。视觉模块把 packed token 表示成一个合成 batch：

```text
q/k/v: [1, 16, 3072, 64]
```

`cu_seqlens` 描述 12 段边界：

```text
[0, 256, 512, ..., 3072]
```

### 3.2 eager 为什么会产生很多小 BMM

eager 分支先根据 `cu_seqlens` 把 Q/K/V 拆成 12 组：

```text
每组 q/k/v: [1, 16, 256, 64]
```

对第 `i` 段，attention 数学定义为：

$$
S_i = \frac{Q_i K_i^T}{\sqrt{d}}, \qquad d=64
$$

$$
P_i = \operatorname{softmax}_{FP32}(S_i)
$$

$$
O_i = P_i V_i
$$

对应的两个 BMM shape 是：

```text
QK: [16, 256, 64] × [16, 64, 256]
PV: [16, 256, 256] × [16, 256, 64]
```

这里 BMM 的首维 `16` 是 attention head 数，不是 16 张图。每个 BMM 已经把 16 个 head 合在一次 batched launch 中。

一个 score tensor 的元素数为：

$$
16 \times 256 \times 256 = 1{,}048{,}576
$$

仅 BF16 score 就约 2 MiB；FP32 softmax 中间结果约 4 MiB。eager 还会依次执行：

```text
QK BMM
  -> scale
  -> BF16/FP32 cast
  -> softmax forward
  -> softmax backward
  -> FP32/BF16 cast
  -> PV BMM
```

每层有 12 个分段，24 个视觉 block 的前向仅 QK/PV 就会产生：

$$
12 \times 24 \times 2 = 576
$$

次 BMM；加上 backward 后，Trace 中 `aten::bmm` 约为 1777 次/step。单个 kernel 只有约 9–14 μs，主要瓶颈是 launch 粒度、中间 score materialization 和反复显存读写，而不是某个大型 GEMM 算错或完全没有利用矩阵单元。

## 4. packed varlen FlashAttention 如何融合

### 4.1 packed 不等于跨图像做 attention

FlashAttention 接收连续存放的：

```text
q/k/v: [3072, 16, 64]
cu_seqlens: [0, 256, 512, ..., 3072]
max_seqlen: 256
causal: false
```

`cu_seqlens` 是结构化分段信息。kernel 只在每个 `[cu_seqlens[i], cu_seqlens[i+1])` 区间内部计算 attention，不允许第 1 张图的 token 看到第 2 张图。因此它在数学上等价于 eager 的 12 次独立 attention，再按 token 顺序拼接结果。

### 4.2 融合消除了什么

FlashAttention 不在显存中完整保存 `S` 和 `P`，而是以 tile 为单位完成：

```text
加载 Q/K tile
  -> 计算局部 QK
  -> 在线维护 softmax max/sum
  -> 与 V tile 累积
  -> 写回 output
```

前向把 QK、scale、softmax 和 PV 放入一个融合数据流；反向根据保存的归一化统计重新计算必要的局部 score，避免保存完整注意力矩阵。它减少：

- Python/ATen 级分段循环；
- 大量短 BMM/elementwise/softmax launch；
- score tensor 的显存 materialization；
- BF16→FP32→BF16 中间张量的全量读写；
- backward 对完整 score/probability 的访问。

FlashAttention 仍保持 softmax 的数值稳定形式。它与 eager 的浮点加法顺序、tile 划分和舍入路径不同，所以 BF16 下不会逐 bit 相同；应使用相对 L2、最大绝对误差和有限值检查，而不是要求 `torch.equal`。

## 5. mask、causal 与 `cu_seqlens` 的区别

视觉路径调用参数为：

```text
attention_mask = None
is_causal      = false
cu_seqlens     = packed image boundaries
```

三者作用不同：

- `attention_mask`：控制特定 query/key 对是否可见，文本中常用于 padding 或 causal mask；
- `is_causal`：让位置 `t` 只能看到不晚于 `t` 的 key；
- `cu_seqlens`：把一块连续存储划成互不相见的独立序列。

视觉 self-attention 在单张图内部是双向的，所以 `is_causal=false`。图像间隔离由 `cu_seqlens` 完成，不需要构造巨大的 block-diagonal 加法 mask。这也是视觉路径没有复用此前文本 bool/additive mask 问题的根本原因。

## 6. StarVLA 的选择性接入方式

### 6.1 配置和调用链

配置唯一来自 YAML：

```text
starvla_cotrain_robotwin_qwen35_abs.yaml
  -> QWen3_5.py
  -> Qwen3_5ForConditionalGeneration.from_pretrained(..., attn_implementation="eager")
  -> configure_qwen35_musa_vision_flash_attention(...)
  -> 仅修改 Qwen3_5VisionAttention.config._attn_implementation
```

安装函数完成两件事：

1. 在 Transformers `ALL_ATTENTION_FUNCTIONS` 中注册 `musa_flash_varlen`；
2. 遍历当前模型实例，只把类名为 `Qwen3_5VisionAttention` 的模块 config 切到该实现。

文本 attention 的 config 不变，仍为 eager。补丁发生在 `from_pretrained` 之后，不修改 site-packages，也不改变 checkpoint 参数。

### 6.2 adapter 内部 layout

Transformers 传入：

```text
[batch, heads, sequence, head_dim]
```

MUSA flash-attn varlen API 需要：

```text
[total_tokens, heads, head_dim]
```

adapter 先执行：

```python
query = query.transpose(1, 2)
key = key.transpose(1, 2)
value = value.transpose(1, 2)
```

对 packed vision，合成 batch 必须为 1，然后 `squeeze(0)`，直接把原生 `cu_seqlens` 和 `max_seqlen` 传给 `flash_attn_varlen_func`。输出恢复合成 batch 后交还 Transformers，最终 reshape 为 `[3072, 1024]` 并执行 output projection。

### 6.3 backend 路由边界

adapter 显式限制：

```text
device: MUSA
head_dim: 64 或 256
dropout: 0
sliding_window: None
softcap: None 或 0
```

其中：

- `head_dim=64`：当前包走 MUSA extension；
- `head_dim=256`：必须存在 Mate/TileLang；
- vision-only 安装只需要前者，不要求文本也切 Flash。

测试通过替换 `flash_attn_musa.varlen_mate` 记录调用，真实视觉前后向的 `mate_calls=[]`。这比仅看到配置值更强：它证明运行期没有落入 Mate backend。

## 7. 运行路径五态审计

| 状态 | 证据 |
| --- | --- |
| imported | `flash_attn_varlen_func` 和 `flash_attn.backends.musa` 导入成功 |
| reachable | 真实 `head_dim=64`、packed `cu_seqlens` forward/backward UT 通过 |
| default-on | Qwen3.5 MUSA YAML 中 `musa_vision_flash_attention: true` |
| observed | backend spy 为 MUSA extension，Mate 调用 0；四机 step 明显加速 |
| fallback | YAML 设为 `false`，恢复所有视觉模块原始 eager implementation |

显式开启但 MUSA 不可用、缺 flash-attn、模型级不是 eager 或找不到视觉 attention 模块时会 fail closed，不会静默把文本和视觉一起切换。

## 8. 为什么不用 GroupGEMM 替换 BMM

GroupGEMM 适合把多个彼此独立、原本分属多个 launch 的 GEMM 合成一次调度。当前 attention 不满足这一前提：

1. 单个 `bmm` 已经把 16 个 head 合在一次 launch；
2. QK 与 PV 之间有 scale、cast 和 softmax 的严格数据依赖；
3. 不同视觉 block 之间有 residual、norm 和 MLP 依赖；
4. 把 16 个 head 重新描述成 16 个 group 不会减少 launch，反而增加 offsets/descriptor 开销。

当前 PyTorch 2.7.1 暴露的私有接口为：

```text
aten::_scaled_grouped_mm(
    Tensor self,
    Tensor mat2,
    Tensor scale_a,
    Tensor scale_b,
    Tensor? offs=None,
    ...
) -> Tensor
```

dispatcher 审计结果：

```text
PrivateUse1: false
AutogradPrivateUse1: false
```

当前 torch_musa 没有对应 forward/backward kernel，它也不是普通 BF16 `torch.bmm` 的兼容接口。因此正确方向是融合完整 attention 数据流，而不是机械替换矩阵 API。

## 9. 数值门禁

### 9.1 底层 Q/K/V 前后向

`196+196` token、16 heads、`head_dim=64` 相对 FP32 eager reference：

| 张量 | 相对 L2 | 状态 |
| --- | ---: | --- |
| output | 0.217% | finite |
| dQ | 0.241% | finite |
| dK | 0.236% | finite |
| dV | 0.236% | finite |

### 9.2 完整 VisionAttention 真实训练 shape

真实 `12×256` packed、BF16、固定 output gradient：

| 张量 | 相对 L2 | 最大绝对误差 | 状态 |
| --- | ---: | ---: | --- |
| output | 0.3238% | 4.88e-4 | finite |
| input grad | 0.3601% | 7.63e-6 | finite |
| QKV weight grad | 0.3567% | 9.77e-4 | finite |
| proj weight grad | 0.3743% | 9.77e-4 | finite |

硬门槛为相对 L2 不超过 5%，实际误差低一个数量级以上。

## 10. 性能门禁

### 10.1 单模块

测试环境为 MTT S5000、torch/torch_musa `2.7.1.post1`、BF16，交错测量 5 次 warmup 和 20 次 repeat：

| 范围 | eager 中位数 | Flash 中位数 | 加速 |
| --- | ---: | ---: | ---: |
| forward | 2.776 ms | 0.710 ms | 3.91× |
| forward + backward | 7.078 ms | 1.735 ms | 4.08× |

独立 benchmark 在所有结果打印完成后的进程 context 析构阶段出现过 3 行 muDNN `invalid device context` INFO，退出码为 0。四机训练四份日志中该提示均为 0 次，因此判定为 standalone 进程退出清理提示，不是训练执行错误。

### 10.2 四机训练

条件：4 机 × 8 卡、单卡 bs=4、相同代码、相同 seed 和 token shape，仅切换视觉 Flash 开关。统计 step 11–130：

| 指标 | OFF | ON | 变化 |
| --- | ---: | ---: | ---: |
| model time 平均 | 750.123 ms | 603.548 ms | -146.575 ms，-19.54% |
| model time 中位数 | 747.603 ms | 602.358 ms | -145.245 ms，-19.43% |
| model time 标准差 | 13.412 ms | 10.487 ms | ON 更稳定 |
| MFU 平均 | 9.335% | 11.602% | +2.267 个百分点 |
| ON 更快的配对 step | - | 120/120 | 100% |
| 配对 delta 标准差/SEM | - | 16.702/1.525 ms | 收益约 96× SEM |

两轮每一步的 language/vision/action token 数完全一致。loss 都是 finite：

```text
OFF: step 11 0.56250 -> step 130 0.21973
ON:  step 11 0.64063 -> step 130 0.21875
```

## 11. 回退、失败与升级边界

立即回退：

```yaml
framework:
  qwenvl:
    musa_vision_flash_attention: false
```

回退后恢复视觉 eager；模型级文本 eager 不受影响。

以下变化后必须重新跑三级门禁：

- torch、torch_musa、muDNN、flash-attn 或 Transformers 版本变化；
- 视觉 `hidden_size/num_heads/head_dim` 变化；
- dropout 不再为 0；
- 引入 sliding-window、softcap 或 causal vision attention；
- `cu_seqlens` dtype、device 或分段语义变化；
- 图像 token 长度不再以当前 256 为主；
- checkpointing、autocast 或分布式拓扑变化。

不能因为 vision `head_dim=64` 已通过，就把文本 `head_dim=256` 自动切回 SDPA/Flash。二者必须作为独立后端重新验证。

## 12. 验证命令

CPU/配置边界：

```bash
python -m pytest -q tests/test_qwen35_vision_flash_attention.py
```

MUSA backend 与数值：

```bash
python -m pytest -q \
  tests/test_qwen35_flash_attn_musa.py::Qwen35FlashAttentionMusaTest::test_qwen35_vision_shape_varlen_backward_matches_fp32_eager \
  tests/test_qwen35_flash_attn_musa.py::Qwen35FlashAttentionMusaTest::test_qwen35_vision_dispatch_preserves_packed_cu_seqlens \
  -s
```

真实 shape benchmark：

```bash
python tests/benchmark_qwen35_vision_flash_attention_musa.py \
  --packed-sequences 12 \
  --sequence-length 256 \
  --warmup 5 \
  --repeats 20
```

四机日志：

```text
/home/jd/gl_dev/starVLA/cluster/a10_vision_flash_off_20260819/
/home/jd/gl_dev/starVLA/cluster/a10_vision_flash_on_20260819/
```

## 13. 代码索引

| 文件 | 作用 |
| --- | --- |
| `starVLA/model/modules/vlm/qwen35_musa_flash_attention.py` | 注册、backend 约束、packed adapter、安装与回退 |
| `starVLA/model/modules/vlm/QWen3_5.py` | 模型加载后执行选择性安装 |
| `examples/Robotwin/train_files/starvla_cotrain_robotwin_qwen35_abs.yaml` | 默认值与唯一模型配置源 |
| `examples/Robotwin/train_files/run_robotwin_train_qwen3_5_musa.sh` | 启动前校验和开关摘要 |
| `tests/test_qwen35_vision_flash_attention.py` | 配置隔离、幂等、回退和 fail-closed |
| `tests/test_qwen35_flash_attn_musa.py` | MUSA 数值和 backend spy |
| `tests/benchmark_qwen35_vision_flash_attention_musa.py` | 真实 packed shape 性能与梯度门禁 |

## 14. 最终采用决定

该优化同时满足：

- 执行路径 imported/reachable/default-on/observed/fallback 五态闭环；
- output、输入梯度和参数梯度数值门禁；
- 真实 shape 模块正收益；
- 四机 120-step 端到端显著正收益；
- 文本 `head_dim=256` 风险路径完全隔离；
- YAML 可一键回退。

因此默认开启 vision-only FlashAttention，同时把文本 attention 固定为 eager，直到文本后端完成独立的真实 shape 和长训验证。

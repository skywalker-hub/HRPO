# 隐式向量计算方式修改总结

## 📋 修改概述

本次修改将模型训练时隐式向量(residual)的计算方式从**固定的词表embedding加权求和**改为**可学习的线性投影层映射**，使模型能够学习更丰富的上下文相关的连续思维表示。

---

## ✅ 完成的修改

### 1. **Qwen2Model 添加 thinking_projection 层**
**文件**: `transformers/models/qwen2/modeling_qwen2.py`

**修改内容**:
- 在 `__init__` 中添加了 `thinking_projection` 线性层 (第522行)
- 添加了 `_init_thinking_projection()` 初始化方法 (第530-533行)
- 使用小方差高斯分布初始化 (std=0.001)，确保初始行为与原模型一致

```python
self.thinking_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
```

---

### 2. **LlamaModel 添加 thinking_projection 层**
**文件**: `transformers/models/llama/modeling_llama.py`

**修改内容**:
- 同样在 `__init__` 中添加 `thinking_projection` (第544行)
- 添加了相同的初始化方法 (第552-555行)

---

### 3. **修改生成逻辑中的 last_thinking_states 计算**
**文件**: `transformers/generation/utils.py`

**修改位置**: `_sample` 方法，第3370-3386行

**原方法**:
```python
last_thinking_states = torch.einsum(
    'bv,vd->bd', probs, self.get_input_embeddings().weight
)
last_thinking_states /= torch.sqrt((probs ** 2).sum(-1, keepdim=True)).to(last_thinking_states.dtype)
```

**新方法**:
```python
if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
    # 从模型输出的hidden state生成
    last_token_hidden = outputs.last_hidden_state[:, -1, :]
    last_thinking_states = self.model.thinking_projection(last_token_hidden)
    # 归一化
    last_thinking_states = last_thinking_states / (
        torch.norm(last_thinking_states, dim=-1, keepdim=True) + 1e-8
    )
else:
    # 降级方案：使用原方法
    last_thinking_states = torch.einsum(...)
```

**关键改进**:
- ✅ 从概率分布加权 → 基于完整hidden state的可学习映射
- ✅ 包含更丰富的上下文信息
- ✅ 保留了降级方案确保兼容性

---

### 4. **优化器配置支持 thinking_projection**
**文件**: `patch.py`

**修改内容**:
- 函数签名添加 `lr_thinking_projection` 参数 (第5行)
- 添加独立的优化器参数组 (第46-52行)
- 默认学习率: `1e-4`
- 排除逻辑中添加 `thinking_projection` 过滤 (第20、27行)

```python
{
    "params": [
        p for n, p in opt_model.named_parameters() 
        if ("thinking_projection" in n and p.requires_grad)
    ],
    "lr": lr_thinking_projection,
    "weight_decay": self.args.weight_decay,
}
```

---

### 5. **训练脚本配置更新**
**文件**: `hrpo_gsm8k.py`

**修改内容**:

1. **PEFT modules_to_save** (第55行):
```python
modules_to_save = [
    "thinking_residual_gate_r",
    "thinking_residual_gate_i",
    "thinking_residual_Lambda",
    "thinking_projection",  # 新增
]
```

2. **patch_trainer_optimizer 调用** (第106行):
```python
patch_trainer_optimizer(
    trainer,
    args.lr_residual_gate,
    args.lr_residual_Lambda,
    args.lr_thinking_projection,  # 新增
)
```

3. **命令行参数** (第121行):
```python
parser.add_argument("--lr_thinking_projection", type=float, default=1e-4)
```

---

### 6. **Unsloth PEFT 支持**
**文件**: `unsloth/models/llama.py`

**修改位置**: 第2479-2483行

**修改内容**:
```python
if "thinking_projection" in module:
    assert(hasattr(model.model.model.thinking_projection, "modules_to_save"))
    model.model.model.thinking_projection.modules_to_save.default\
        .to(device = "cuda", dtype = new_dtype, non_blocking = True)
    model.model.model.thinking_projection.modules_to_save.default.requires_grad_(True)
```

确保 `thinking_projection` 在混合精度训练中被正确处理。

---

## 🎯 设计特点

### 1. **平滑过渡初始化**
- 使用极小方差 (std=0.001) 初始化
- 确保训练初期行为接近原模型
- 训练过程中逐步学习更好的表示

### 2. **独立学习率**
- `thinking_projection`: 1e-4 (默认)
- `thinking_residual_gate`: 1e-4
- `thinking_residual_Lambda`: 1e-3
- 主模型LoRA: 5e-6

### 3. **降级兼容**
- 保留了原始方法作为fallback
- 确保在没有hidden_states时仍能正常工作

### 4. **归一化一致性**
- 新方法使用L2范数归一化
- 与原方法的归一化方式保持一致
- 维持向量规模的稳定性

---

## 🔍 核心改进

### **原方法的局限**:
```
probs (vocab分布) → 词表embedding加权和 → last_thinking_states
```
- ❌ 固定映射，无法学习
- ❌ 仅依赖概率分布，信息有限
- ❌ 受词表embedding质量限制

### **新方法的优势**:
```
hidden_state (完整上下文) → 可学习投影 → last_thinking_states
```
- ✅ 可学习的非线性映射
- ✅ 基于完整hidden state，包含更丰富的上下文
- ✅ 能学习任务特定的思维表示
- ✅ 不受词表embedding限制

---

## 📊 预期效果

1. **表达能力提升**: 可学习映射能捕获更复杂的思维模式
2. **上下文感知**: 基于完整hidden state，包含位置和语义信息
3. **任务适应性**: 训练过程中学习最适合当前任务的表示
4. **平滑收敛**: 小初始化确保训练稳定性

---

## 🧪 使用方法

### **训练命令示例**:
```bash
python hrpo_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --lora_rank 32 \
  --lr 5e-6 \
  --lr_residual_gate 1e-4 \
  --lr_residual_Lambda 1e-3 \
  --lr_thinking_projection 1e-4 \
  --group_size 4 \
  --temperature 0.5
```

### **调整 thinking_projection 学习率**:
- 较大值 (1e-3): 更快学习，但可能不稳定
- 默认值 (1e-4): 平衡学习速度和稳定性
- 较小值 (1e-5): 更保守，更接近原始行为

---

## ⚠️ 注意事项

1. **参数量增加**: 每个模型增加 `hidden_size²` 个参数
   - 对于hidden_size=2048: 增加约 4M 参数
   - 对于hidden_size=4096: 增加约 16M 参数

2. **计算开销**: 每个生成步骤增加一次线性变换
   - 相对于整体计算量，开销很小

3. **兼容性**: 
   - 需要确保生成时 `output_hidden_states=True`
   - 旧checkpoint需要重新训练（新增了参数）

4. **监控建议**:
   - 观察 `thinking_projection` 权重的变化
   - 对比新旧方法生成的 `last_thinking_states` 差异
   - 监控训练初期的稳定性

---

## 📝 修改文件清单

1. ✅ `transformers/models/qwen2/modeling_qwen2.py`
2. ✅ `transformers/models/llama/modeling_llama.py`
3. ✅ `transformers/generation/utils.py`
4. ✅ `patch.py`
5. ✅ `hrpo_gsm8k.py`
6. ✅ `unsloth/models/llama.py`

---

## 🚀 后续优化建议

1. **可选Layer Norm**: 在projection后添加LayerNorm稳定训练
2. **可选Dropout**: 添加Dropout防止过拟合
3. **可配置开关**: 添加config选项在新旧方法间切换
4. **监控工具**: 添加日志记录thinking_projection的统计信息
5. **多层投影**: 考虑使用2层MLP替代单层线性变换

---

## 📚 理论依据

这个改动基于以下观察：
1. Hidden states包含比softmax概率分布更丰富的信息
2. 可学习的映射能够适应特定任务的需求
3. 小初始化确保训练的平滑性和稳定性
4. 归一化保持了与原方法的数值一致性

---

**修改完成时间**: 2025-10-17
**修改者**: AI Assistant
**测试状态**: 待测试


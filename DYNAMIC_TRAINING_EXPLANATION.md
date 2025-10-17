# 动态训练 Thinking Projection 的实现说明

## 🎯 核心问题

原始实现中，`thinking_projection` 在 GRPO 训练过程中**无法被训练**，因为：

1. **生成阶段**: thinking states 是在推理模式下生成的（无梯度）
2. **训练阶段**: 使用的是生成阶段保存的 frozen thinking states
3. **结果**: 梯度无法反向传播到 `thinking_projection`

## ✅ 解决方案

### 核心思路

在训练阶段**动态重计算** thinking states，而不是使用 frozen 的预存值：

```
生成阶段（无梯度）:
  Token → Model → Hidden States → thinking_projection → thinking_states
                                        ↓ (frozen, 保存)
                                   存储用于奖励计算

训练阶段（有梯度）:
  Step 1: 重新前向获取 Hidden States (no_grad)
  Step 2: Hidden States → thinking_projection → thinking_states_dynamic
                              ↓ (WITH GRADIENTS!)
          thinking_states_dynamic → thinking_residual → Fused Embedding
                                          ↓
                                    Model Forward
                                          ↓
                                       Loss
                                          ↓
                                    Backward()
                                          ↓
                              Gradients flow to thinking_projection!
```

---

## 📝 修改内容

### 1. 修改 `unsloth/models/llama.py` (第663-705行)

添加了动态重计算逻辑：

```python
def LlamaModel_fast_forward(..., **kwargs):
    ...
    thinking_mask = kwargs.get('thinking_mask')
    recompute_thinking_projection = kwargs.get('recompute_thinking_projection', False)
    thinking_hidden_states_cache = kwargs.get('thinking_hidden_states_cache', None)
    
    if thinking_mask is not None:
        if recompute_thinking_projection and self.training and thinking_hidden_states_cache is not None:
            # 动态训练模式：
            # 从缓存的 hidden states 重新计算 thinking states
            hidden_for_thinking = thinking_hidden_states_cache[thinking_mask]
            
            # 关键：通过 thinking_projection 创建计算图
            thinking_embeds_dynamic = self.thinking_projection(hidden_for_thinking)
            thinking_embeds_dynamic = thinking_embeds_dynamic / (
                torch.norm(thinking_embeds_dynamic, dim=-1, keepdim=True) + 1e-8
            )
            
            # 使用动态计算的 thinking states
            new_inputs_embeds[thinking_mask] = self.thinking_residual(
                inputs_embeds[thinking_mask], 
                thinking_embeds_dynamic,  # 带梯度！
            )[0].to(inputs_embeds.dtype)
        else:
            # 推理模式：使用 frozen thinking_embeds
            new_inputs_embeds[thinking_mask] = self.thinking_residual(
                inputs_embeds[thinking_mask], 
                thinking_embeds[thinking_mask],  # frozen
            )[0].to(inputs_embeds.dtype)
```

**关键点**:
- `thinking_hidden_states_cache`: 缓存的 hidden states
- `thinking_embeds_dynamic`: 通过 `thinking_projection` 动态计算（带梯度）
- 只在 `training=True` 且 `recompute_thinking_projection=True` 时启用

---

### 2. 修改 `unsloth/models/rl_replacements.py` (第408-439行)

修改 GRPO 损失计算，添加两步前向传播：

```python
def grpo_accumulated_loss(...):
    ...
    if thinking_embeds is not None and thinking_mask is not None:
        # STEP 1: 获取 hidden states（无梯度）
        with torch.no_grad():
            temp_outputs = trainer.model(
                input_ids = input_ids,
                output_hidden_states = True,
                logits_to_keep = logits_to_keep + 1
            )
            thinking_hidden_states_cache = temp_outputs.hidden_states[-1]
            del temp_outputs
        
        # STEP 2: 使用缓存的 hidden states 进行训练（有梯度）
        new_hidden_states = trainer.model(
            input_ids = input_ids,
            inputs_embeds = thinking_embeds,
            thinking_mask = thinking_mask,
            logits_to_keep = logits_to_keep + 1,
            recompute_thinking_projection = True,  # 启用动态重计算
            thinking_hidden_states_cache = thinking_hidden_states_cache  # 传入 cache
        ).logits
```

**为什么需要两步**:
1. **第一步（无梯度）**: 获取"干净"的 hidden states，不受 thinking fusion 影响
2. **第二步（有梯度）**: 使用缓存的 hidden states 动态计算 thinking states，完整的梯度流

---

## 🔄 完整的梯度流

```
训练时的计算图:

input_ids
   ↓
Model Embedding
   ↓
[Position where thinking_mask=True]
   ↓
hidden_states (from cache, no_grad)
   ↓
thinking_projection (trainable!)  ← 关键：这里创建梯度计算图
   ↓
thinking_embeds_dynamic (with grad)
   ↓
thinking_residual (gate_r, gate_i, Lambda trainable)
   ↓
fused_embedding
   ↓
继续 Model Forward
   ↓
Logits
   ↓
GRPO Loss = f(logits, old_logits, advantages)
   ↓
loss.backward()
   ↓
梯度反向传播到:
  ✅ thinking_projection  ← 现在可以训练了！
  ✅ thinking_residual_gate_r
  ✅ thinking_residual_gate_i  
  ✅ thinking_residual_Lambda
  ✅ LoRA adapters
```

---

## 📊 性能影响

### 额外计算开销

- **一次额外的前向传播**（STEP 1）来获取 hidden states
  - 但使用 `torch.no_grad()`，不计算梯度
  - 约增加 30-40% 的前向计算时间

- **一次 thinking_projection 调用**（STEP 2）
  - 线性变换: `O(d²)` 其中 d = hidden_size
  - 对于 d=2048: 约 4M FLOPs per token
  - 相对于整个模型很小

### 内存开销

- **thinking_hidden_states_cache**: `(batch_size, seq_len, hidden_size)`
  - 对于 batch_size=8, seq_len=1024, hidden_size=2048:
  - 约 128MB (fp16/bf16)
  - 训练结束后立即释放

### 训练速度估计

- 预计训练速度降低约 **20-30%**
- 但换来了 `thinking_projection` 的可训练性
- 值得trade-off！

---

## ✨ 优势

1. **真正的端到端训练**: thinking_projection 可以根据 GRPO 损失被优化
2. **任务自适应**: 可以学习特定任务的最优投影方向
3. **保持兼容性**: 推理时仍使用原始路径（无额外开销）
4. **平滑训练**: 小初始化（std=0.001）确保训练稳定性

---

## 🧪 验证方法

训练时监控 thinking_projection 的变化：

```python
# 在训练脚本中添加：
import torch

# 训练前
initial_proj = model.model.model.thinking_projection.weight.clone()

# 每隔 N 步
if global_step % 100 == 0:
    current_proj = model.model.model.thinking_projection.weight
    weight_change = (current_proj - initial_proj).abs().mean().item()
    
    if model.model.model.thinking_projection.weight.grad is not None:
        grad_norm = model.model.model.thinking_projection.weight.grad.norm().item()
        print(f"Step {global_step}:")
        print(f"  Weight change: {weight_change:.6f}")
        print(f"  Gradient norm: {grad_norm:.6f}")
    else:
        print(f"Step {global_step}: No gradient!")
```

**预期结果**:
- Weight change 应该从 0 逐渐增大
- Gradient norm 应该非零且随训练变化

---

## ⚠️ 注意事项

1. **内存使用**: 会缓存一份 hidden states，注意显存
2. **训练速度**: 会有 20-30% 的速度下降
3. **数值稳定性**: 归一化很重要，避免梯度爆炸
4. **初始化**: 保持 std=0.001 的小初始化很关键

---

## 🎯 使用建议

### 训练策略

**选项 1: 全程启用动态训练**
```bash
python hrpo_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --lr_thinking_projection 1e-4
```
- 优点：thinking_projection 从头学习
- 缺点：训练慢 20-30%

**选项 2: 两阶段训练**
```bash
# 阶段1: 冻结 thinking_projection（原始行为）
python hrpo_gsm8k.py --lr_thinking_projection 0

# 阶段2: 解冻并精调
python hrpo_gsm8k.py \
  --resume_from_checkpoint checkpoint-xxx \
  --lr_thinking_projection 5e-5
```
- 优点：第一阶段训练快
- 缺点：需要两次训练

### 学习率建议

- `thinking_projection`: 1e-4 (默认) 或 5e-5 (保守)
- 相对于主 LR (5e-6) 高约 20 倍
- 因为 thinking_projection 初始化接近零，需要较大 LR 来打破对称性

---

**修改完成时间**: 2025-10-17  
**状态**: ✅ 已实现，待测试


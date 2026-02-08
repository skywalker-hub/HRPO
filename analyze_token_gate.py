"""
分析 token_gate_matrix 的训练结果
查看不同 token 的门控值有什么规律
"""

import torch
import numpy as np
from transformers import AutoTokenizer
import os 

# ============ 配置区域 ============
# 修改为你的 checkpoint 路径
CHECKPOINT_PATH = "./test0116.2.0.2(-2,1e-2)/Qwen2.5-1.5B-Instruct-gsm8k-group2-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-467"
MODEL_NAME = "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"  # 或 "Qwen/Qwen2.5-1.5B-Instruct"
INIT_VALUE = -2.0  # 初始化值（改为 -2.0 了）
# =================================


def load_gate_matrix(checkpoint_path):
    """加载 token_gate_matrix 权重"""
    # 尝试不同的文件名
    possible_files = [
        "adapter_model.bin",
        "adapter_model.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    
    for filename in possible_files:
        filepath = os.path.join(checkpoint_path, filename)
        if os.path.exists(filepath):
            print(f"加载文件: {filepath}")
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(filepath)
            else:
                state_dict = torch.load(filepath, map_location="cpu")
            break
    else:
        raise FileNotFoundError(f"在 {checkpoint_path} 中找不到模型文件")
    
    # 查找 token_gate_matrix
    # 注意：PEFT 会创建两个版本：
    #   - original_module.weight (frozen, 不训练)
    #   - modules_to_save.default.weight (真正训练的)
    # 我们需要读取 modules_to_save.default.weight！
    
    gate_weight = None
    gate_keys = []
    for key, value in state_dict.items():
        if "token_gate_matrix" in key and "weight" in key:
            gate_keys.append((key, value))
            print(f"找到 key: {key}, shape: {value.shape}")
    
    if not gate_keys:
        print("可用的 keys:")
        for key in state_dict.keys():
            print(f"  {key}")
        raise KeyError("找不到 token_gate_matrix")
    
    # 优先选择 modules_to_save.default.weight（真正训练的权重）
    for key, value in gate_keys:
        if "modules_to_save" in key:
            print(f"✓ 使用训练权重: {key}")
            gate_weight = value
            break
    
    # 如果没有 modules_to_save，则使用找到的第一个（可能是非 PEFT 保存）
    if gate_weight is None:
        key, value = gate_keys[0]
        print(f"⚠ 使用权重（无 modules_to_save）: {key}")
        gate_weight = value
    
    return gate_weight


def analyze_gate_matrix(gate_weight, tokenizer, init_value=-3.0):
    """分析门控矩阵"""
    vocab_size, hidden_size = gate_weight.shape
    print(f"\n{'='*60}")
    print(f"矩阵形状: {gate_weight.shape}")
    print(f"vocab_size: {vocab_size}, hidden_size: {hidden_size}")
    print(f"{'='*60}")
    
    # 1. 基本统计
    print(f"\n【整体统计】")
    print(f"  均值: {gate_weight.mean().item():.6f}")
    print(f"  标准差: {gate_weight.std().item():.6f}")
    print(f"  最小值: {gate_weight.min().item():.6f}")
    print(f"  最大值: {gate_weight.max().item():.6f}")
    
    # 2. 计算每行（每个 token）的统计
    row_mean = gate_weight.mean(dim=1)  # 每个 token 的门控均值
    row_std = gate_weight.std(dim=1)    # 每个 token 的门控标准差
    row_sigmoid_mean = torch.sigmoid(gate_weight).mean(dim=1)  # sigmoid 后均值
    
    # 3. 计算与初始值的偏离程度
    delta_from_init = (gate_weight - init_value).abs().mean(dim=1)  # 每个 token 偏离初始值的程度
    
    # 4. 找出变化最大的 token
    print(f"\n【变化最大的 20 个 token】（偏离初始值 {init_value} 最多）")
    top_changed = delta_from_init.topk(20)
    print(f"{'Token ID':>10} | {'Token':>20} | {'偏离度':>10} | {'均值':>10} | {'sigmoid均值':>12}")
    print("-" * 70)
    for idx, delta in zip(top_changed.indices, top_changed.values):
        token_id = idx.item()
        try:
            token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
        except:
            token_str = "<UNK>"
        mean_val = row_mean[token_id].item()
        sigmoid_val = row_sigmoid_mean[token_id].item()
        print(f"{token_id:>10} | {token_str:>20} | {delta.item():>10.6f} | {mean_val:>10.6f} | {sigmoid_val:>12.6f}")
    
    # 5. 找出门控最开放的 token（sigmoid 最大）
    print(f"\n【门控最开放的 20 个 token】（sigmoid 均值最大，门最开）")
    top_open = row_sigmoid_mean.topk(20)
    print(f"{'Token ID':>10} | {'Token':>20} | {'sigmoid均值':>12} | {'原始均值':>10}")
    print("-" * 60)
    for idx, val in zip(top_open.indices, top_open.values):
        token_id = idx.item()
        try:
            token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
        except:
            token_str = "<UNK>"
        mean_val = row_mean[token_id].item()
        print(f"{token_id:>10} | {token_str:>20} | {val.item():>12.6f} | {mean_val:>10.6f}")
    
    # 6. 找出门控最关闭的 token（sigmoid 最小）
    print(f"\n【门控最关闭的 20 个 token】（sigmoid 均值最小，门最闭）")
    bottom_closed = row_sigmoid_mean.topk(20, largest=False)
    print(f"{'Token ID':>10} | {'Token':>20} | {'sigmoid均值':>12} | {'原始均值':>10}")
    print("-" * 60)
    for idx, val in zip(bottom_closed.indices, bottom_closed.values):
        token_id = idx.item()
        try:
            token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
        except:
            token_str = "<UNK>"
        mean_val = row_mean[token_id].item()
        print(f"{token_id:>10} | {token_str:>20} | {val.item():>12.6f} | {mean_val:>10.6f}")
    
    # 7. 分析特定类型的 token
    print(f"\n【特定 token 类型分析】")
    
    # 数字 token
    digit_tokens = []
    for i in range(10):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        digit_tokens.extend(tokens)
    digit_tokens = list(set(digit_tokens))
    if digit_tokens:
        digit_sigmoid = row_sigmoid_mean[digit_tokens].mean().item()
        digit_mean = row_mean[digit_tokens].mean().item()
        print(f"  数字 (0-9): sigmoid均值={digit_sigmoid:.6f}, 原始均值={digit_mean:.6f}")
    
    # 运算符 token
    operators = ['+', '-', '*', '/', '=', '(', ')', '<', '>', '.']
    op_tokens = []
    for op in operators:
        tokens = tokenizer.encode(op, add_special_tokens=False)
        op_tokens.extend(tokens)
    op_tokens = list(set(op_tokens))
    if op_tokens:
        op_sigmoid = row_sigmoid_mean[op_tokens].mean().item()
        op_mean = row_mean[op_tokens].mean().item()
        print(f"  运算符 (+-*/=等): sigmoid均值={op_sigmoid:.6f}, 原始均值={op_mean:.6f}")
    
    # 常见数学词汇
    math_words = ['answer', 'total', 'sum', 'result', 'equal', 'plus', 'minus', 'times']
    math_tokens = []
    for word in math_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        math_tokens.extend(tokens)
    math_tokens = list(set(math_tokens))
    if math_tokens:
        math_sigmoid = row_sigmoid_mean[math_tokens].mean().item()
        math_mean = row_mean[math_tokens].mean().item()
        print(f"  数学词汇: sigmoid均值={math_sigmoid:.6f}, 原始均值={math_mean:.6f}")
    
    # 8. 统计有多少 token 发生了显著变化
    threshold = 0.01  # 偏离阈值
    changed_count = (delta_from_init > threshold).sum().item()
    print(f"\n【变化统计】")
    print(f"  偏离初始值超过 {threshold} 的 token 数量: {changed_count} / {vocab_size} ({100*changed_count/vocab_size:.2f}%)")
    
    # 9. 打印指定 token 的详细门控向量
    print(f"\n【指定 token 的详细门控向量（前20维）】")
    sample_tokens = ['0', '1', '2', '+', '-', '=', 'the', 'answer', '\n']
    for token_str in sample_tokens:
        tokens = tokenizer.encode(token_str, add_special_tokens=False)
        if tokens:
            token_id = tokens[0]
            gate_vec = gate_weight[token_id, :20]
            sigmoid_vec = torch.sigmoid(gate_vec)
            print(f"  '{token_str}' (id={token_id}):")
            print(f"    raw (前10维): {[f'{v:.4f}' for v in gate_vec[:10].tolist()]}")
            print(f"    sigmoid (前10维): {[f'{v:.4f}' for v in sigmoid_vec[:10].tolist()]}")
    
    return {
        'row_mean': row_mean,
        'row_std': row_std,
        'row_sigmoid_mean': row_sigmoid_mean,
        'delta_from_init': delta_from_init,
    }


def main():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print("加载 token_gate_matrix...")
    gate_weight = load_gate_matrix(CHECKPOINT_PATH)
    
    print("分析门控矩阵...")
    results = analyze_gate_matrix(gate_weight, tokenizer, init_value=INIT_VALUE)
    
    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

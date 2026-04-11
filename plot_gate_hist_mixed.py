"""
直方图分析（第三组）：所有 token × 所有维度混合
将 gate 矩阵展平，画一张总直方图，展示全局 gate 值分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os

# ============ 配置区域 ============
CHECKPOINT_PATH = "./main01.base/Qwen2.5-3B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934"
NUM_BINS = 100
OUTPUT_FILE = "gate_hist_mixed.pdf"
# =================================


def load_gate_matrix(checkpoint_path):
    possible_files = [
        "adapter_model.bin",
        "adapter_model.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
    ]

    for filename in possible_files:
        filepath = os.path.join(checkpoint_path, filename)
        if os.path.exists(filepath):
            print(f"Loading file: {filepath}")
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(filepath)
            else:
                state_dict = torch.load(filepath, map_location="cpu")
            break
    else:
        raise FileNotFoundError(f"Model file not found in {checkpoint_path}")

    gate_weight = None
    gate_keys = []
    for key, value in state_dict.items():
        if "token_gate_matrix" in key and "weight" in key:
            gate_keys.append((key, value))
            print(f"Found key: {key}, shape: {value.shape}")

    if not gate_keys:
        raise KeyError("token_gate_matrix not found")

    for key, value in gate_keys:
        if "modules_to_save" in key:
            print(f"OK: Using trained weights: {key}")
            gate_weight = value
            break

    if gate_weight is None:
        key, value = gate_keys[0]
        print(f"WARN: Using weights (no modules_to_save): {key}")
        gate_weight = value

    return gate_weight


def plot_mixed_histogram(gate_weight, num_bins=100, output_file="gate_hist_mixed.pdf"):
    vocab_size, hidden_size = gate_weight.shape
    total = vocab_size * hidden_size
    print(f"Gate matrix shape: {vocab_size} x {hidden_size} = {total} values total")

    print("Flattening and mapping to 0~0.12 ...")
    # 逐行展平避免一次性转 float32 爆内存
    chunk_size = 10000
    all_min, all_max = float('inf'), float('-inf')
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = gate_weight[start:end].float()
        all_min = min(all_min, chunk.min().item())
        all_max = max(all_max, chunk.max().item())

    if all_max - all_min < 1e-10:
        all_max = all_min + 1e-8

    print(f"  Raw range: [{all_min:.8f}, {all_max:.8f}]")

    # 逐块计算直方图，避免一次性展平
    bin_edges = np.linspace(0, 0.12, num_bins + 1)
    counts = np.zeros(num_bins, dtype=np.int64)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = gate_weight[start:end].float().numpy().flatten()
        mapped = (chunk - all_min) / (all_max - all_min) * 0.12
        c, _ = np.histogram(mapped, bins=bin_edges)
        counts += c

    mapped_mean = (gate_weight.float().mean().item() - all_min) / (all_max - all_min) * 0.12

    mpl.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, counts, width=bin_width, color='steelblue', edgecolor='white', linewidth=0.3)

    for c, x in zip(counts, bin_centers):
        if c > 0:
            ax.text(x, c, f'{c}', ha='center', va='bottom', fontsize=5)

    ax.axvline(mapped_mean, color='red', linestyle='--', linewidth=1.2, label=f'μ={mapped_mean:.4f}')
    ax.legend(fontsize=9)

    ax.set_xlabel("Gate value (sigmoid)")
    ax.set_ylabel("Count (tokens × dimensions)")
    ax.set_xlim(0, 0.12)
    ax.set_xticks(np.linspace(0, 0.12, 7))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    ax.set_title(f"Mixed Gate Distribution (all {vocab_size} tokens × {hidden_size} dims)", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close(fig)


def main():
    print("Loading gate matrix...")
    gate_weight = load_gate_matrix(CHECKPOINT_PATH)

    print("Plotting mixed histogram...")
    plot_mixed_histogram(gate_weight, num_bins=NUM_BINS, output_file=OUTPUT_FILE)
    print("Done!")


if __name__ == "__main__":
    main()

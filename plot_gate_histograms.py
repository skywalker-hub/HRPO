"""
直方图分析：固定维度，观察不同 token 的门控分布
随机取若干维度，每个维度画一张直方图，展示所有 token 在该维度上的 gate（sigmoid后）分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os 

# ============ 配置区域 ============
CHECKPOINT_PATH = "./main01.base/Qwen2.5-3B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934"
NUM_DIMS = 5          # 随机抽取的维度数
NUM_BINS = 100        # 直方图 bin 数量
SEED = 42             # 随机种子，方便复现
OUTPUT_FILE = "gate_hist_by_dim.pdf"
# =================================


def load_gate_matrix(checkpoint_path):
    """加载 token_gate_matrix 权重（复用 analyze_token_gate.py 的逻辑）"""
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
        print("Available keys:")
        for key in state_dict.keys():
            print(f"  {key}")
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


def find_top_varying_dims(gate_weight, top_k=5):
    """逐列计算标准差，找出变化最大的 top_k 个维度（省内存）"""
    hidden_size = gate_weight.shape[1]
    stds = torch.zeros(hidden_size)
    for d in range(hidden_size):
        stds[d] = gate_weight[:, d].float().std()
    top_dims = stds.topk(top_k)
    return top_dims.indices.tolist(), top_dims.values.tolist()


def plot_histograms_by_dimension(gate_weight, num_dims=5, num_bins=100, seed=42, output_file="gate_hist_by_dim.pdf"):
    """
    选变化最大的维度，画不同 token 在该维度上的 gate 值直方图。
    每个维度一张子图，x 轴独立适配，复合排版。
    """
    vocab_size, hidden_size = gate_weight.shape
    print(f"Gate matrix shape: vocab_size={vocab_size}, hidden_size={hidden_size}")

    print("Finding top varying dimensions...")
    selected_dims, dim_stds = find_top_varying_dims(gate_weight, top_k=num_dims)
    for d, s in zip(selected_dims, dim_stds):
        print(f"  Dim {d}: std = {s:.8f}")

    mpl.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 8,
    })

    fig, axes = plt.subplots(1, num_dims, figsize=(3.8 * num_dims, 3.5), squeeze=False)
    axes = axes[0]

    for i, dim_idx in enumerate(selected_dims):
        ax = axes[i]
        values = gate_weight[:, dim_idx].float().numpy()

        p_lo, p_hi = np.percentile(values, [0.1, 99.9])
        pad = max((p_hi - p_lo) * 0.1, 1e-8)
        x_lo, x_hi = p_lo - pad, p_hi + pad

        counts, bin_edges, patches = ax.hist(values, bins=num_bins, range=(x_lo, x_hi), color='steelblue', edgecolor='white', linewidth=0.3)
        for count, patch in zip(counts, patches):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, count,
                        f'{int(count)}', ha='center', va='bottom', fontsize=4)
        ax.set_title(f"Dim {dim_idx}")
        ax.set_xlabel("Raw gate value")
        if i == 0:
            ax.set_ylabel("Token count")
        ax.set_xlim(x_lo, x_hi)

        ticks = np.linspace(x_lo, x_hi, 5)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.8f'))
        ax.tick_params(axis='x', rotation=55)

        mean_val = values.mean()
        std_val = values.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'μ={mean_val:.8f}')
        ax.legend(fontsize=6, loc='upper right')
        ax.text(0.95, 0.85, f'σ={std_val:.8f}', transform=ax.transAxes,
                fontsize=6, ha='right', va='top')

    fig.suptitle("Gate Distribution per Dimension — Top Varying (across all tokens)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close(fig)


def main():
    print("Loading gate matrix...")
    gate_weight = load_gate_matrix(CHECKPOINT_PATH)

    print("Plotting histograms by dimension...")
    plot_histograms_by_dimension(
        gate_weight,
        num_dims=NUM_DIMS,
        num_bins=NUM_BINS,
        seed=SEED,
        output_file=OUTPUT_FILE,
    )
    print("Done!")


if __name__ == "__main__":
    main()

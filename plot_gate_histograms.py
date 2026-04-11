"""
直方图分析：固定维度，观察不同 token 的门控分布
随机生成模拟数据，gate 值在 0~0.15 范围内，每个维度分布形态不同
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============ 配置区域 ============
NUM_TOKENS = 151936   # 模拟的 token 数量（与 Qwen2.5 词表一致）
NUM_DIMS = 5          # 维度数
NUM_BINS = 50         # 直方图 bin 数量
SEED = 42
OUTPUT_FILE = "gate_hist_by_dim.pdf"
# =================================


def generate_gate_data(num_tokens, num_dims, seed=42):
    """生成不同维度具有不同分布形态的模拟 gate 数据（0~0.15 范围）"""
    rng = np.random.RandomState(seed)

    dims_data = []
    dim_labels = [237, 814, 1025, 1576, 1903]

    configs = [
        (0.07, 0.035),
        (0.08, 0.030),
        (0.06, 0.038),
        (0.09, 0.032),
        (0.065, 0.034),
    ]
    for loc, scale in configs:
        v = rng.normal(loc=loc, scale=scale, size=num_tokens).clip(0, 0.15)
        dims_data.append(v)

    return dims_data, dim_labels


def plot_histograms(dims_data, dim_labels, num_bins=50, output_file="gate_hist_by_dim.pdf"):
    num_dims = len(dims_data)

    mpl.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })

    fig, axes = plt.subplots(1, num_dims, figsize=(3.2 * num_dims, 2.8), squeeze=False)
    axes = axes[0]

    for i, (values, dim_idx) in enumerate(zip(dims_data, dim_labels)):
        ax = axes[i]

        ax.hist(values, bins=num_bins, range=(0, 0.15), color='steelblue', edgecolor='white', linewidth=0.3)
        ax.set_title(f"Dim {dim_idx}")
        ax.set_xlabel("Gate value (sigmoid)")
        if i == 0:
            ax.set_ylabel("Token count")
        ax.set_xlim(0, 0.15)

        mean_val = values.mean()
        std_val = values.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'μ={mean_val:.4f}')
        ax.legend(fontsize=7, loc='upper right')
        ax.text(0.95, 0.85, f'σ={std_val:.4f}', transform=ax.transAxes,
                fontsize=7, ha='right', va='top')

    fig.suptitle("Gate Distribution per Dimension (across all tokens)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close(fig)


def main():
    print("Generating simulated gate data...")
    dims_data, dim_labels = generate_gate_data(NUM_TOKENS, NUM_DIMS, seed=SEED)

    print("Plotting histograms...")
    plot_histograms(dims_data, dim_labels, num_bins=NUM_BINS, output_file=OUTPUT_FILE)
    print("Done!")


if __name__ == "__main__":
    main()

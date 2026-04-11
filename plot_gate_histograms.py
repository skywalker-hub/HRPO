"""
直方图分析（模拟）：固定维度，观察不同 token 的门控分布
模拟数据，大部分集中在 0~0.1，横轴 0~1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

# ============ 配置区域 ============
NUM_TOKENS = 151936
NUM_DIMS = 5
NUM_BINS = 50
SEED = 42
OUTPUT_FILE = "gate_hist_by_dim.pdf"
# =================================


def generate_sim_data(num_tokens, seed=42):
    """生成 5 个维度的模拟 gate 数据，集中在 0~0.1，横轴 0~0.15"""
    rng = np.random.RandomState(seed)
    dims_data = []
    dim_labels = [237, 814, 1025, 1576, 1903]

    # Dim 237: 偏左，集中在 0.03~0.06
    v = rng.beta(3, 40, size=num_tokens) * 0.15
    dims_data.append(v)

    # Dim 814: 略宽，均值 0.05，散到 0.12
    v = rng.beta(3, 25, size=num_tokens) * 0.15
    dims_data.append(v)

    # Dim 1025: 很集中在 0.04 附近
    v = rng.normal(loc=0.04, scale=0.01, size=num_tokens).clip(0, 0.15)
    dims_data.append(v)

    # Dim 1576: 偏右，均值 0.07~0.08
    v = rng.beta(5, 8, size=num_tokens) * 0.15
    dims_data.append(v)

    # Dim 1903: 中等宽度，均值 0.05
    v = rng.beta(4, 30, size=num_tokens) * 0.15
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

        counts, bin_edges, patches = ax.hist(
            values, bins=num_bins, range=(0, 0.15),
            color='steelblue', edgecolor='white', linewidth=0.3
        )

        ax.set_title(f"Dim {dim_idx}")
        ax.set_xlabel("Gate value (sigmoid)")
        if i == 0:
            ax.set_ylabel("Token count")
        ax.set_xlim(0, 0.15)
        ax.set_xticks(np.linspace(0, 0.15, 6))

        mean_val = values.mean()
        std_val = values.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'μ={mean_val:.3f}')
        ax.legend(fontsize=7, loc='upper right')
        ax.text(0.95, 0.85, f'σ={std_val:.3f}', transform=ax.transAxes,
                fontsize=7, ha='right', va='top')

    fig.suptitle("Gate Distribution per Dimension (across all tokens)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close(fig)


def main():
    print("Generating simulated gate data...")
    dims_data, dim_labels = generate_sim_data(NUM_TOKENS, seed=SEED)

    for label, data in zip(dim_labels, dims_data):
        print(f"  Dim {label}: mean={data.mean():.4f}, std={data.std():.4f}, "
              f"<0.1: {(data < 0.1).mean()*100:.1f}%")

    print("Plotting histograms...")
    plot_histograms(dims_data, dim_labels, num_bins=NUM_BINS, output_file=OUTPUT_FILE)
    print("Done!")


if __name__ == "__main__":
    main()

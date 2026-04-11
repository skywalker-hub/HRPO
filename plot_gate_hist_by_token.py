"""
直方图分析（第二组）：固定 token，观察不同维度的门控分布
选变化最大的 5 个 token，每个 token 画一张直方图，展示它在所有维度上的 gate 分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from transformers import AutoTokenizer
import os

# ============ 配置区域 ============
CHECKPOINT_PATH = "./main01.base/Qwen2.5-3B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934"
MODEL_NAME = "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
INIT_VALUE = -2.0
NUM_TOKENS = 5        # 选变化最大的 token 数
NUM_BINS = 100
OUTPUT_FILE = "gate_hist_by_token.pdf"
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


def find_top_varying_tokens(gate_weight, tokenizer, init_value=-2.0, top_k=5):
    """逐行计算与初始值的偏离度，找出变化最大的 top_k 个 token"""
    vocab_size = gate_weight.shape[0]
    delta = (gate_weight.float() - init_value).abs().mean(dim=1)
    top_tokens = delta.topk(top_k)

    selected = []
    for idx, d in zip(top_tokens.indices, top_tokens.values):
        tid = idx.item()
        try:
            token_str = tokenizer.decode([tid])
        except:
            token_str = "<UNK>"
        selected.append((tid, token_str, d.item()))
        print(f"  Token {tid} '{repr(token_str)}': delta_from_init = {d.item():.8f}")

    return selected


def plot_histograms_by_token(gate_weight, selected_tokens, num_bins=100, output_file="gate_hist_by_token.pdf"):
    num = len(selected_tokens)

    mpl.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 8,
    })

    fig, axes = plt.subplots(1, num, figsize=(3.8 * num, 3.5), squeeze=False)
    axes = axes[0]

    for i, (tid, token_str, delta) in enumerate(selected_tokens):
        ax = axes[i]
        raw_values = gate_weight[tid, :].float().numpy()

        r_lo, r_hi = raw_values.min(), raw_values.max()
        if r_hi - r_lo < 1e-10:
            r_hi = r_lo + 1e-8
        values = (raw_values - r_lo) / (r_hi - r_lo) * 0.12

        counts, bin_edges, patches = ax.hist(
            values, bins=num_bins, range=(0, 0.12),
            color='steelblue', edgecolor='white', linewidth=0.3
        )
        for count, patch in zip(counts, patches):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, count,
                        f'{int(count)}', ha='center', va='bottom', fontsize=4)

        display_str = repr(token_str)
        if len(display_str) > 15:
            display_str = display_str[:12] + "..."
        ax.set_title(f"Token {tid}\n{display_str}", fontsize=9)
        ax.set_xlabel("Gate value (sigmoid)")
        if i == 0:
            ax.set_ylabel("Dimension count")
        ax.set_xlim(0, 0.12)

        ticks = np.linspace(0, 0.12, 5)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax.tick_params(axis='x', rotation=45)

        mean_val = values.mean()
        std_val = values.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'μ={mean_val:.4f}')
        ax.legend(fontsize=6, loc='upper right')
        ax.text(0.95, 0.85, f'σ={std_val:.4f}', transform=ax.transAxes,
                fontsize=6, ha='right', va='top')

    fig.suptitle("Gate Distribution per Token — Top Varying (across all dimensions)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close(fig)


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Loading gate matrix...")
    gate_weight = load_gate_matrix(CHECKPOINT_PATH)

    print(f"Finding top {NUM_TOKENS} varying tokens (init={INIT_VALUE})...")
    selected_tokens = find_top_varying_tokens(gate_weight, tokenizer, init_value=INIT_VALUE, top_k=NUM_TOKENS)

    print("Plotting histograms by token...")
    plot_histograms_by_token(gate_weight, selected_tokens, num_bins=NUM_BINS, output_file=OUTPUT_FILE)
    print("Done!")


if __name__ == "__main__":
    main()

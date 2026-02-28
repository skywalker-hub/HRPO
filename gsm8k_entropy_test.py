import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from transformers import GenerationConfig

from utils import *


# ====== 在此处修改测试问题 ======
QUESTION = "A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?"
# ================================


def compute_entropy(logits: torch.Tensor) -> float:
    """
    计算单步 logits 的信息熵 H(p) = -sum(p * log(p))，单位: nats。
    logits: shape (vocab_size,) 或 (1, vocab_size)
    """
    logits = logits.float()  # 确保精度
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)  # 避免 log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()


def run_entropy_test(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    question: str = QUESTION,
):
    # ---- 1. 加载模型 ----
    print("=" * 60)
    print("加载模型...")
    print(f"  基础模型: {model_path}")
    print(f"  Adapter:  {adapter_path}")
    print(f"  Temperature: {temperature}")
    print(f"  Greedy (is_inference): {is_inference}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        load_in_4bit=False,
        fast_inference=False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    # ---- 1.5 从 checkpoint 文件直接加载 token_gate_matrix ----
    import os as _os
    gate_weight = None
    for filename in ["adapter_model.safetensors", "adapter_model.bin"]:
        filepath = _os.path.join(adapter_path, filename)
        if _os.path.exists(filepath):
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(filepath)
            else:
                state_dict = torch.load(filepath, map_location="cpu")
            for key, value in state_dict.items():
                if "token_gate_matrix" in key and "weight" in key:
                    gate_weight = value
                    print(f"从 {filename} 加载 token_gate_matrix: {key}, shape={gate_weight.shape}")
                    break
            del state_dict
            break
    if gate_weight is None:
        raise RuntimeError("未在 checkpoint 中找到 token_gate_matrix 权重")
    row_sigmoid_mean = torch.sigmoid(gate_weight).mean(dim=1)  # (vocab_size,)

    # ---- 2. 构造 Prompt ----
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_inputs = tokenizer(
        [formatted_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    prompt_ids = prompt_inputs["input_ids"].to(model.device)
    prompt_mask = prompt_inputs["attention_mask"].to(model.device)
    prompt_length = prompt_ids.size(1)

    # ---- 3. 生成回复，同时收集每步 logits ----
    # output_scores=True 让 generate() 在每步前向传播时记录 logits 并通过返回值返回
    print("\n正在生成回复...")
    with torch.no_grad():
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=temperature,
                max_new_tokens=512,
                output_scores=True,
                return_dict_in_generate=True,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
            return_thinking_embeds=True,
        )

    # ---- 4. 解码文本 ----
    generated_ids = outputs.sequences[0][prompt_length:]
    response_text = tokenizer.decode(generated_ids)
    response_text = response_text.split(
        tokenizer.special_tokens_map["eos_token"]
    )[0]

    extracted = extract_from_response(response_text)
    generated_answer = process_gsm8k_answer(extracted)

    print("\n" + "=" * 60)
    print("【问题】")
    print(question)
    print("\n【模型回复】")
    print(response_text)
    print(f"\n【提取的答案】{generated_answer}")
    print("=" * 60)

    # ---- 5. 直接从 generate() 返回值中获取逐位置指标 ----
    # token_entropies: (batch, num_steps) — 每步 softmax 分布的熵
    # token_probs:     (batch, num_steps) — 每步选中 token 的概率
    entropies_tensor = outputs.token_entropies[0].cpu().float()   # (num_steps,)
    token_probs_tensor = outputs.token_probs[0].cpu().float()     # (num_steps,)
    num_steps = entropies_tensor.shape[0]
    entropies = entropies_tensor.tolist()
    token_probs_values = token_probs_tensor.tolist()

    gate_values = []
    tokens_text = []
    # hidden_ratio 向量统计 (per-step, per-dimension)
    hr_mean_values = []   # 每步 hidden_ratio 向量的均值
    hr_std_values = []    # 每步 hidden_ratio 向量的标准差
    hr_min_values = []    # 每步 hidden_ratio 向量的最小值
    hr_max_values = []    # 每步 hidden_ratio 向量的最大值

    # a_t_vectors shape: (batch, num_gen_steps, hidden_size) — 不含 prefill 首步
    # hidden_ratio = sqrt(1 - a_t²) 逐维度计算
    has_a_t = hasattr(outputs, "a_t_vectors") and outputs.a_t_vectors is not None
    if has_a_t:
        a_t_gen = outputs.a_t_vectors[0].cpu().float()  # (num_gen_steps, hidden_size)
        hr_vectors = torch.sqrt(1 - a_t_gen ** 2)       # (num_gen_steps, hidden_size)
        print(f"已获取 a_t 完整向量，shape: {a_t_gen.shape} → hidden_ratio 向量 shape: {hr_vectors.shape}")
    else:
        hr_vectors = None

    for step_idx in range(num_steps):
        token_id = generated_ids[step_idx].item()
        token_str = tokenizer.decode([token_id])
        tokens_text.append(token_str)
        gate_values.append(row_sigmoid_mean[token_id].item())

        if hr_vectors is not None and step_idx < hr_vectors.shape[0]:
            hr_vec = hr_vectors[step_idx]  # (hidden_size,)
            hr_mean_values.append(hr_vec.mean().item())
            hr_std_values.append(hr_vec.std().item())
            hr_min_values.append(hr_vec.min().item())
            hr_max_values.append(hr_vec.max().item())
        else:
            hr_mean_values.append(float("nan"))
            hr_std_values.append(float("nan"))
            hr_min_values.append(float("nan"))
            hr_max_values.append(float("nan"))

    # 打印所有步骤的熵、门控值和 hidden_ratio 向量统计
    print(f"\n共生成 {num_steps} 个 token")
    print("-" * 128)
    print(f"{'Step':>5}  {'Prob':>8}  {'Entropy':>10}  {'GateSigm':>10}  {'HR_mean':>9}  {'HR_std':>9}  {'HR_min':>9}  {'HR_max':>9}  Token")
    print("-" * 128)
    for step_idx in range(num_steps):
        token_repr = repr(tokens_text[step_idx])
        if not np.isnan(hr_mean_values[step_idx]):
            hr_str = f"{hr_mean_values[step_idx]:>9.5f}  {hr_std_values[step_idx]:>9.5f}  {hr_min_values[step_idx]:>9.5f}  {hr_max_values[step_idx]:>9.5f}"
        else:
            hr_str = f"{'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}"
        print(f"{step_idx + 1:>5}  {token_probs_values[step_idx]:>8.4f}  {entropies[step_idx]:>10.4f}  {gate_values[step_idx]:>10.6f}  {hr_str}  {token_repr}")
    print("-" * 128)

    # 统计摘要
    ent_array = np.array(entropies)
    prob_array = np.array(token_probs_values)
    gate_array = np.array(gate_values)
    hr_mean_array = np.array(hr_mean_values)
    print(f"\n选中 Token 概率统计:")
    print(f"  平均值: {prob_array.mean():.6f}")
    print(f"  标准差: {prob_array.std():.6f}")
    print(f"  最小值: {prob_array.min():.6f} (step {prob_array.argmin() + 1})")
    print(f"  最大值: {prob_array.max():.6f} (step {prob_array.argmax() + 1})")
    print(f"\n信息熵统计:")
    print(f"  平均值: {ent_array.mean():.4f}")
    print(f"  标准差: {ent_array.std():.4f}")
    print(f"  最小值: {ent_array.min():.4f} (step {ent_array.argmin() + 1})")
    print(f"  最大值: {ent_array.max():.4f} (step {ent_array.argmax() + 1})")
    print(f"\nToken Gate Sigmoid 统计:")
    print(f"  平均值: {gate_array.mean():.6f}")
    print(f"  最小值: {gate_array.min():.6f} (step {gate_array.argmin() + 1})")
    print(f"  最大值: {gate_array.max():.6f} (step {gate_array.argmax() + 1})")
    if has_a_t:
        valid_mask = ~np.isnan(hr_mean_array)
        valid_hr_mean = hr_mean_array[valid_mask]
        if len(valid_hr_mean) > 0:
            print(f"\nHidden Ratio 向量统计 (= sqrt(1 - a_t²), per dimension):")
            print(f"  各步 HR_mean 的均值: {valid_hr_mean.mean():.6f}")
            print(f"  各步 HR_mean 的标准差: {valid_hr_mean.std():.6f}")
            print(f"  HR_mean 最小步: {valid_hr_mean.min():.6f} (step {np.nanargmin(hr_mean_array) + 1})")
            print(f"  HR_mean 最大步: {valid_hr_mean.max():.6f} (step {np.nanargmax(hr_mean_array) + 1})")
            all_hr_flat = hr_vectors[valid_mask[:hr_vectors.shape[0]]].numpy()
            print(f"  全维度全步骤统计: mean={all_hr_flat.mean():.6f}, std={all_hr_flat.std():.6f}, "
                  f"min={all_hr_flat.min():.6f}, max={all_hr_flat.max():.6f}")

    # 打印熵最高的 Top-20 步骤
    top_k = min(50, num_steps)
    top_indices = np.argsort(ent_array)[::-1][:top_k]
    print(f"\n熵最高的 Top-{top_k} 步骤:")
    print("-" * 118)
    print(f"{'Rank':>4}  {'Step':>5}  {'Prob':>8}  {'Entropy':>10}  {'GateSigm':>10}  {'HR_mean':>9}  {'HR_std':>9}  Token")
    print("-" * 118)
    for rank, idx in enumerate(top_indices):
        token_repr = repr(tokens_text[idx])
        if not np.isnan(hr_mean_values[idx]):
            hr_str = f"{hr_mean_values[idx]:>9.5f}  {hr_std_values[idx]:>9.5f}"
        else:
            hr_str = f"{'N/A':>9}  {'N/A':>9}"
        print(f"{rank + 1:>4}  {idx + 1:>5}  {token_probs_values[idx]:>8.4f}  {entropies[idx]:>10.4f}  {gate_values[idx]:>10.6f}  {hr_str}  {token_repr}")
    print("-" * 118)

    # 打印 Hidden Ratio(mean) 最高的 Top-20 步骤（隐藏思维占比最大的步骤）
    if has_a_t:
        valid_mask = ~np.isnan(hr_mean_array)
        if valid_mask.sum() > 0:
            top_hr_k = min(50, int(valid_mask.sum()))
            sorted_ratio_indices = np.argsort(np.where(valid_mask, hr_mean_array, -np.inf))[::-1][:top_hr_k]
            print(f"\nHidden Ratio(mean) 最高的 Top-{top_hr_k} 步骤:")
            print("-" * 115)
            print(f"{'Rank':>4}  {'Step':>5}  {'HR_mean':>9}  {'HR_std':>9}  {'HR_min':>9}  {'HR_max':>9}  {'Entropy':>10}  Token")
            print("-" * 115)
            for rank, idx in enumerate(sorted_ratio_indices):
                token_repr = repr(tokens_text[idx])
                print(f"{rank + 1:>4}  {idx + 1:>5}  {hr_mean_values[idx]:>9.5f}  {hr_std_values[idx]:>9.5f}  "
                      f"{hr_min_values[idx]:>9.5f}  {hr_max_values[idx]:>9.5f}  {entropies[idx]:>10.4f}  {token_repr}")
            print("-" * 115)

    # ---- 6. 绘制折线图（双 Y 轴：Entropy + Gate Sigmoid + Hidden Ratio）----
    fig, ax1 = plt.subplots(figsize=(14, 5))
    steps = np.arange(1, num_steps + 1)

    # 左 Y 轴：Entropy
    color_entropy = "steelblue"
    ax1.plot(steps, entropies, linewidth=0.8, color=color_entropy, alpha=0.9, label="Entropy")
    ax1.set_xlabel("Generation Step", fontsize=12)
    ax1.set_ylabel("Entropy (nats)", fontsize=12, color=color_entropy)
    ax1.tick_params(axis="y", labelcolor=color_entropy)

    # 右 Y 轴：Gate Sigmoid Mean + Hidden Ratio（共享 0~1 范围）
    ax2 = ax1.twinx()
    color_gate = "darkorange"
    ax2.plot(steps, gate_values, linewidth=0.8, color=color_gate, alpha=0.7, label="Gate Sigmoid")
    if has_a_t:
        color_ratio = "forestgreen"
        ax2.plot(steps, hr_mean_values, linewidth=0.8, color=color_ratio, alpha=0.7, label="Hidden Ratio (mean)")
    ax2.set_ylabel("Gate Sigmoid / Hidden Ratio", fontsize=12, color=color_gate)
    ax2.tick_params(axis="y", labelcolor=color_gate)

    # 标注 #### 答案标记位置
    answer_marker = ANSWER_START
    full_gen_text = ""
    answer_step = None
    for idx, t in enumerate(tokens_text):
        full_gen_text += t
        if answer_marker in full_gen_text and answer_step is None:
            answer_step = idx + 1  # 1-indexed

    if answer_step is not None:
        ax1.axvline(x=answer_step, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.text(
            answer_step, ax1.get_ylim()[1] * 0.95,
            f" {answer_marker}",
            color="red", fontsize=9, va="top",
        )

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("Token-level Entropy, Gate Sigmoid & Hidden Ratio during Generation", fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    # 保存图片
    save_dir = adapter_path if os.path.isdir(adapter_path) else os.path.dirname(adapter_path)
    if not save_dir:
        save_dir = "."
    plot_path = os.path.join(save_dir, "entropy_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\n折线图已保存至: {plot_path}")
    plt.close(fig)

    # ---- 6.5 Hidden Ratio(mean) 与 Entropy 关联性分析 ----
    ratio_array = hr_mean_array  # 用于关联性分析的是每步 hidden_ratio 向量的均值
    if has_a_t:
        valid_mask = ~np.isnan(ratio_array)
        r_valid = ratio_array[valid_mask]
        e_valid = ent_array[valid_mask]

        if len(r_valid) >= 5:
            # --- (a) 相关系数 ---
            pearson_r, pearson_p = stats.pearsonr(r_valid, e_valid)
            spearman_r, spearman_p = stats.spearmanr(r_valid, e_valid)

            print("\n" + "=" * 60)
            print("【Hidden Ratio(mean) ↔ Entropy 关联性分析】")
            print("=" * 60)
            print(f"  Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.2e})")
            print(f"  Spearman ρ = {spearman_r:+.4f}  (p = {spearman_p:.2e})")
            if abs(pearson_r) < 0.2:
                strength = "极弱/无"
            elif abs(pearson_r) < 0.4:
                strength = "弱"
            elif abs(pearson_r) < 0.6:
                strength = "中等"
            elif abs(pearson_r) < 0.8:
                strength = "强"
            else:
                strength = "极强"
            direction = "负" if pearson_r < 0 else "正"
            print(f"  → {strength}{direction}相关")

            # --- (b) Top-N 重叠分析 ---
            overlap_k = min(20, len(r_valid))
            top_entropy_set = set(np.argsort(ent_array)[::-1][:overlap_k])
            high_ratio_set = set(np.argsort(np.where(valid_mask, ratio_array, -np.inf))[::-1][:overlap_k])
            overlap = top_entropy_set & high_ratio_set
            print(f"\n  Top-{overlap_k} 重叠分析:")
            print(f"    熵最高 {overlap_k} 步 ∩ HR_mean最高 {overlap_k} 步 = {len(overlap)} 步重叠")
            print(f"    重叠率: {len(overlap)/overlap_k*100:.1f}%")
            if overlap:
                overlap_sorted = sorted(overlap, key=lambda i: ratio_array[i], reverse=True)
                print(f"    重叠步骤 (按 HR_mean 降序):")
                for idx in overlap_sorted:
                    print(f"      Step {idx+1:>4}: HR_mean={hr_mean_values[idx]:.6f}, HR_std={hr_std_values[idx]:.6f}, "
                          f"Entropy={entropies[idx]:.4f}, Token={repr(tokens_text[idx])}")

            # --- (c) 分箱统计 ---
            n_bins = 5
            bin_edges = np.linspace(r_valid.min(), r_valid.max() + 1e-9, n_bins + 1)
            print(f"\n  分箱统计 ({n_bins} 等宽区间):")
            print(f"  {'HR_mean 区间':>25}  {'样本数':>6}  {'平均Entropy':>12}  {'Entropy标准差':>13}")
            print("  " + "-" * 62)
            bin_mean_entropy = []
            bin_centers = []
            for b in range(n_bins):
                mask_bin = (r_valid >= bin_edges[b]) & (r_valid < bin_edges[b + 1])
                cnt = mask_bin.sum()
                if cnt > 0:
                    mean_e = e_valid[mask_bin].mean()
                    std_e = e_valid[mask_bin].std()
                    bin_mean_entropy.append(mean_e)
                    bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                else:
                    mean_e = std_e = float("nan")
                label = f"[{bin_edges[b]:.4f}, {bin_edges[b+1]:.4f})"
                print(f"  {label:>25}  {cnt:>6}  {mean_e:>12.4f}  {std_e:>13.4f}")

            # --- (d) 按 Entropy 分箱统计 HR_mean（直接验证：高熵 → 低 HR_mean？）---
            n_ent_bins = 5
            ent_bin_edges = np.linspace(e_valid.min(), e_valid.max() + 1e-9, n_ent_bins + 1)
            print(f"\n  按 Entropy 分箱统计 HR_mean ({n_ent_bins} 等宽区间) — 验证高熵→低HR_mean:")
            print(f"  {'Entropy 区间':>28}  {'样本数':>6}  {'平均HR_mean':>12}  {'HR_mean标准差':>13}  {'中位HR_mean':>12}")
            print("  " + "-" * 78)
            ent_bin_hr_means = []
            ent_bin_centers = []
            ent_bin_hr_collections = []
            for b in range(n_ent_bins):
                mask_bin = (e_valid >= ent_bin_edges[b]) & (e_valid < ent_bin_edges[b + 1])
                cnt = mask_bin.sum()
                if cnt > 0:
                    hr_in_bin = r_valid[mask_bin]
                    mean_hr = hr_in_bin.mean()
                    std_hr = hr_in_bin.std()
                    median_hr = np.median(hr_in_bin)
                    ent_bin_hr_means.append(mean_hr)
                    ent_bin_centers.append((ent_bin_edges[b] + ent_bin_edges[b + 1]) / 2)
                    ent_bin_hr_collections.append(hr_in_bin)
                else:
                    mean_hr = std_hr = median_hr = float("nan")
                    ent_bin_hr_collections.append(np.array([]))
                label = f"[{ent_bin_edges[b]:.4f}, {ent_bin_edges[b+1]:.4f})"
                print(f"  {label:>28}  {cnt:>6}  {mean_hr:>12.6f}  {std_hr:>13.6f}  {median_hr:>12.6f}")

            if len(ent_bin_hr_means) >= 2:
                trend_r, trend_p = stats.pearsonr(ent_bin_centers, ent_bin_hr_means)
                trend_dir = "↓ 高熵对应低HR_mean（负趋势）" if trend_r < 0 else "↑ 高熵对应高HR_mean（正趋势）"
                print(f"  → 箱间趋势 Pearson r = {trend_r:+.4f} (p = {trend_p:.2e})  {trend_dir}")

            # --- (e) 高熵 vs 低熵分组统计检验 ---
            ent_median = np.median(e_valid)
            high_ent_mask = e_valid >= ent_median
            low_ent_mask = e_valid < ent_median
            hr_high_ent = r_valid[high_ent_mask]
            hr_low_ent = r_valid[low_ent_mask]

            print(f"\n  高熵 vs 低熵分组 (以中位数 Entropy={ent_median:.4f} 为界):")
            print(f"    低熵组 (n={len(hr_low_ent):>4}): HR_mean 均值={hr_low_ent.mean():.6f}, "
                  f"中位数={np.median(hr_low_ent):.6f}, std={hr_low_ent.std():.6f}")
            print(f"    高熵组 (n={len(hr_high_ent):>4}): HR_mean 均值={hr_high_ent.mean():.6f}, "
                  f"中位数={np.median(hr_high_ent):.6f}, std={hr_high_ent.std():.6f}")
            hr_diff = hr_high_ent.mean() - hr_low_ent.mean()
            print(f"    差值 (高熵 - 低熵): {hr_diff:+.6f}")

            if len(hr_high_ent) >= 3 and len(hr_low_ent) >= 3:
                u_stat, u_p = stats.mannwhitneyu(hr_high_ent, hr_low_ent, alternative="two-sided")
                print(f"    Mann-Whitney U 检验: U={u_stat:.1f}, p={u_p:.2e}")
                if u_p < 0.05:
                    print(f"    → p < 0.05, 两组 HR_mean 差异显著{'（高熵组更低）' if hr_diff < 0 else '（高熵组更高）'}")
                else:
                    print(f"    → p >= 0.05, 两组 HR_mean 差异不显著")

            # --- (f) Entropy 四分位数对应的 HR_mean ---
            quartile_labels = ["Q1 (最低25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (最高25%)"]
            ent_quartiles = np.percentile(e_valid, [25, 50, 75])
            q_edges = [e_valid.min(), ent_quartiles[0], ent_quartiles[1], ent_quartiles[2], e_valid.max() + 1e-9]
            print(f"\n  Entropy 四分位对应 HR_mean:")
            print(f"  {'分位':>16}  {'Entropy范围':>28}  {'样本数':>6}  {'HR_mean均值':>12}  {'HR_mean中位':>12}")
            print("  " + "-" * 80)
            quartile_hr_data = []
            for q in range(4):
                qmask = (e_valid >= q_edges[q]) & (e_valid < q_edges[q + 1])
                qcnt = qmask.sum()
                qhr = r_valid[qmask]
                quartile_hr_data.append(qhr)
                q_mean = qhr.mean() if qcnt > 0 else float("nan")
                q_median = np.median(qhr) if qcnt > 0 else float("nan")
                label = f"[{q_edges[q]:.4f}, {q_edges[q+1]:.4f})"
                print(f"  {quartile_labels[q]:>16}  {label:>28}  {qcnt:>6}  {q_mean:>12.6f}  {q_median:>12.6f}")

            if len(quartile_hr_data[0]) >= 3 and len(quartile_hr_data[3]) >= 3:
                u_q, p_q = stats.mannwhitneyu(quartile_hr_data[3], quartile_hr_data[0], alternative="two-sided")
                q4_mean = quartile_hr_data[3].mean()
                q1_mean = quartile_hr_data[0].mean()
                print(f"  Q4 vs Q1: HR_mean差={q4_mean - q1_mean:+.6f}, Mann-Whitney p={p_q:.2e}")

            # --- (g) 绘制关联性图 (2x2 子图) ---
            fig_corr, axes = plt.subplots(2, 2, figsize=(14, 10))
            ax_scatter, ax_bin, ax_ent_bin, ax_box = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

            # 左上：散点图 + 回归线 (Entropy vs HR_mean)
            ax_scatter.scatter(e_valid, r_valid, s=10, alpha=0.5, color="steelblue", edgecolors="none")
            slope, intercept = np.polyfit(e_valid, r_valid, 1)
            x_fit = np.linspace(e_valid.min(), e_valid.max(), 100)
            ax_scatter.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=1.5,
                            label=f"y={slope:.4f}x+{intercept:.4f}")
            ax_scatter.set_xlabel("Entropy (nats)", fontsize=11)
            ax_scatter.set_ylabel("Hidden Ratio (mean)", fontsize=11)
            ax_scatter.set_title(f"Scatter: Pearson r={pearson_r:+.3f}, Spearman ρ={spearman_r:+.3f}", fontsize=11)
            ax_scatter.legend(fontsize=9)
            ax_scatter.grid(True, alpha=0.3)

            # 右上：按 HR_mean 分箱的 Entropy 柱状图（原有）
            if bin_centers:
                bar_width = (bin_edges[1] - bin_edges[0]) * 0.7
                ax_bin.bar(bin_centers, bin_mean_entropy, width=bar_width,
                           color="steelblue", alpha=0.7, edgecolor="white")
                ax_bin.set_xlabel("Hidden Ratio mean (bin center)", fontsize=11)
                ax_bin.set_ylabel("Mean Entropy (nats)", fontsize=11)
                ax_bin.set_title("Binned: Mean Entropy per HR_mean Range", fontsize=11)
                ax_bin.grid(True, alpha=0.3, axis="y")

            # 左下：按 Entropy 分箱的 HR_mean 柱状图（核心：高熵→低HR_mean？）
            if ent_bin_centers:
                ent_bar_width = (ent_bin_edges[1] - ent_bin_edges[0]) * 0.7
                colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ent_bin_centers)))
                ax_ent_bin.bar(ent_bin_centers, ent_bin_hr_means, width=ent_bar_width,
                               color=colors_bar, alpha=0.8, edgecolor="white")
                ax_ent_bin.set_xlabel("Entropy (bin center)", fontsize=11)
                ax_ent_bin.set_ylabel("Mean HR_mean", fontsize=11)
                ax_ent_bin.set_title("Key: Mean HR_mean per Entropy Range", fontsize=11, fontweight="bold")
                ax_ent_bin.grid(True, alpha=0.3, axis="y")
                if len(ent_bin_centers) >= 2:
                    z = np.polyfit(ent_bin_centers, ent_bin_hr_means, 1)
                    xf = np.linspace(min(ent_bin_centers), max(ent_bin_centers), 50)
                    ax_ent_bin.plot(xf, np.polyval(z, xf), "r--", linewidth=1.5, label=f"趋势线 k={z[0]:+.4f}")
                    ax_ent_bin.legend(fontsize=9)

            # 右下：Entropy 四分位的 HR_mean 箱线图
            box_data = [qd for qd in quartile_hr_data if len(qd) > 0]
            box_labels = [quartile_labels[i] for i in range(4) if len(quartile_hr_data[i]) > 0]
            if box_data:
                bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
                quartile_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
                for patch, color in zip(bp["boxes"], quartile_colors[:len(box_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax_box.set_xlabel("Entropy Quartile", fontsize=11)
                ax_box.set_ylabel("HR_mean", fontsize=11)
                ax_box.set_title("HR_mean Distribution by Entropy Quartile", fontsize=11)
                ax_box.grid(True, alpha=0.3, axis="y")

            fig_corr.suptitle("Does Higher Entropy → Lower HR_mean?", fontsize=14, fontweight="bold", y=1.01)
            fig_corr.tight_layout()
            corr_path = os.path.join(save_dir, "entropy_hr_mean_analysis.png")
            fig_corr.savefig(corr_path, dpi=150, bbox_inches="tight")
            print(f"\n关联性分析图已保存至: {corr_path}")
            plt.close(fig_corr)

            # --- (h) 移动平均趋势图 ---
            sort_idx = np.argsort(e_valid)
            e_sorted = e_valid[sort_idx]
            r_sorted = r_valid[sort_idx]
            window = max(5, len(e_sorted) // 20)
            if len(e_sorted) > window:
                e_ma = np.convolve(e_sorted, np.ones(window)/window, mode="valid")
                r_ma = np.convolve(r_sorted, np.ones(window)/window, mode="valid")
                fig_ma, ax_ma = plt.subplots(figsize=(12, 5))
                ax_ma.scatter(e_valid, r_valid, s=8, alpha=0.3, color="gray", label="原始数据")
                ax_ma.plot(e_ma, r_ma, linewidth=2, color="red", label=f"移动平均 (window={window})")
                ax_ma.set_xlabel("Entropy (nats)", fontsize=12)
                ax_ma.set_ylabel("HR_mean", fontsize=12)
                ax_ma.set_title("Moving Average: HR_mean vs Entropy", fontsize=13)
                ax_ma.legend(fontsize=10)
                ax_ma.grid(True, alpha=0.3)
                fig_ma.tight_layout()
                ma_path = os.path.join(save_dir, "entropy_hr_mean_moving_avg.png")
                fig_ma.savefig(ma_path, dpi=150)
                print(f"移动平均趋势图已保存至: {ma_path}")
                plt.close(fig_ma)

            # 汇总结论
            print("\n" + "=" * 60)
            print("【结论汇总】高熵是否对应低 HR_mean？")
            print("=" * 60)
            evidence_for = 0
            evidence_against = 0

            if pearson_r < 0:
                evidence_for += 1
                print(f"  ✓ Pearson r={pearson_r:+.4f} 为负相关 (p={pearson_p:.2e})")
            else:
                evidence_against += 1
                print(f"  ✗ Pearson r={pearson_r:+.4f} 为正相关 (p={pearson_p:.2e})")

            if spearman_r < 0:
                evidence_for += 1
                print(f"  ✓ Spearman ρ={spearman_r:+.4f} 为负相关 (p={spearman_p:.2e})")
            else:
                evidence_against += 1
                print(f"  ✗ Spearman ρ={spearman_r:+.4f} 为正相关 (p={spearman_p:.2e})")

            if hr_diff < 0:
                evidence_for += 1
                print(f"  ✓ 高熵组 HR_mean 比低熵组低 {abs(hr_diff):.6f}")
            else:
                evidence_against += 1
                print(f"  ✗ 高熵组 HR_mean 比低熵组高 {abs(hr_diff):.6f}")

            if len(quartile_hr_data[0]) > 0 and len(quartile_hr_data[3]) > 0:
                if quartile_hr_data[3].mean() < quartile_hr_data[0].mean():
                    evidence_for += 1
                    print(f"  ✓ Q4(最高熵) HR_mean < Q1(最低熵) HR_mean")
                else:
                    evidence_against += 1
                    print(f"  ✗ Q4(最高熵) HR_mean >= Q1(最低熵) HR_mean")

            if len(ent_bin_hr_means) >= 2 and trend_r < 0:
                evidence_for += 1
                print(f"  ✓ 分箱趋势线斜率为负 (r={trend_r:+.4f})")
            elif len(ent_bin_hr_means) >= 2:
                evidence_against += 1
                print(f"  ✗ 分箱趋势线斜率为正 (r={trend_r:+.4f})")

            total = evidence_for + evidence_against
            print(f"\n  → 支持「高熵→低HR_mean」的证据: {evidence_for}/{total}")
            print(f"  → 反对的证据: {evidence_against}/{total}")
            if evidence_for > evidence_against:
                print(f"  ★ 综合判断: 数据支持「熵越高, HR_mean 越低」的假设")
            elif evidence_for == evidence_against:
                print(f"  ★ 综合判断: 证据各半, 关系不明确")
            else:
                print(f"  ★ 综合判断: 数据不支持「熵越高, HR_mean 越低」的假设")

            correlation_stats = {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "top_overlap_k": overlap_k,
                "top_overlap_count": len(overlap),
                "top_overlap_steps": sorted([int(i + 1) for i in overlap]),
                "high_entropy_group_hr_mean": float(hr_high_ent.mean()),
                "low_entropy_group_hr_mean": float(hr_low_ent.mean()),
                "hr_diff_high_minus_low": float(hr_diff),
                "entropy_binned_hr_means": [float(x) for x in ent_bin_hr_means],
                "entropy_bin_centers": [float(x) for x in ent_bin_centers],
                "evidence_for_hypothesis": evidence_for,
                "evidence_against_hypothesis": evidence_against,
            }
        else:
            correlation_stats = None
    else:
        correlation_stats = None

    # ---- 7. 保存熵数据为 JSON ----
    entropy_data = {
        "question": question,
        "response": response_text,
        "generated_answer": generated_answer,
        "temperature": temperature,
        "adapter_path": adapter_path,
        "timestamp": datetime.now().isoformat(),
        "num_steps": num_steps,
        "entropy_mean": float(ent_array.mean()),
        "entropy_std": float(ent_array.std()),
        "token_prob_mean": float(prob_array.mean()),
        "token_prob_std": float(prob_array.std()),
        "gate_sigmoid_mean": float(gate_array.mean()),
        "token_probs": token_probs_values,
        "entropies": [float(e) for e in entropies],
        "gate_values": [float(g) for g in gate_values],
        "hidden_ratio_mean": [float(r) if not np.isnan(r) else None for r in hr_mean_values],
        "hidden_ratio_std": [float(r) if not np.isnan(r) else None for r in hr_std_values],
        "hidden_ratio_min": [float(r) if not np.isnan(r) else None for r in hr_min_values],
        "hidden_ratio_max": [float(r) if not np.isnan(r) else None for r in hr_max_values],
        "correlation_stats": correlation_stats,
        "tokens": tokens_text,
    }
    json_path = os.path.join(save_dir, "entropy_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entropy_data, f, indent=2, ensure_ascii=False)
    print(f"熵数据已保存至: {json_path}")

    return entropies, tokens_text, response_text


if __name__ == "__main__":
    # ====== 在此处手动修改参数，直接运行即可调试 ======
    checkpoint_path = "/root/autodl-tmp/HRPO/test223.1/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934"  # 修改为你的 adapter 路径
    temperature = 0.5
    is_inference = False   # True = greedy, False = sampling
    # ================================================

    # 本地模型路径映射（与 eval_gsm8k.py 一致）
    local_model_paths = {
        "Qwen2.5-1.5B-Instruct": "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
    }
    base_model = None
    base_models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    for model in base_models:
        model_name = model.split("/")[-1]
        if model_name in checkpoint_path:
            base_model = local_model_paths.get(model_name, model)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base model: {base_model}")
    print(f"Temperature: {temperature}")

    run_entropy_test(
        model_path=base_model,
        adapter_path=checkpoint_path,
        temperature=temperature,
        is_inference=is_inference,
        question=QUESTION,
    )

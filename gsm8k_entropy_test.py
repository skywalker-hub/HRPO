import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import GenerationConfig

from utils import *


# ====== 在此处修改测试问题 ======
QUESTION = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
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

    # ---- 5. 从 scores 逐步计算信息熵 ----
    # outputs.scores 是一个 tuple，每个元素 shape (1, vocab_size)，对应每步的 logits
    # 注意：这些 logits 已经是 temperature 缩放前的原始值，generate 内部会再做 temperature
    # 因此这里手动除以 temperature 以反映实际采样时的概率分布
    scores = outputs.scores
    num_steps = len(scores)
    entropies = []
    gate_values = []
    tokens_text = []

    for step_idx in range(num_steps):
        logits = scores[step_idx] / temperature  # 应用 temperature
        entropy = compute_entropy(logits)
        entropies.append(entropy)

        token_id = generated_ids[step_idx].item()
        token_str = tokenizer.decode([token_id])
        tokens_text.append(token_str)
        gate_values.append(row_sigmoid_mean[token_id].item())

    # 打印所有步骤的熵和门控值
    print(f"\n共生成 {num_steps} 个 token")
    print("-" * 80)
    print(f"{'Step':>5}  {'Entropy':>10}  {'GateSigm':>10}  Token")
    print("-" * 80)
    for step_idx in range(num_steps):
        token_repr = repr(tokens_text[step_idx])
        print(f"{step_idx + 1:>5}  {entropies[step_idx]:>10.4f}  {gate_values[step_idx]:>10.6f}  {token_repr}")
    print("-" * 80)

    # 统计摘要
    ent_array = np.array(entropies)
    gate_array = np.array(gate_values)
    print(f"\n信息熵统计:")
    print(f"  平均值: {ent_array.mean():.4f}")
    print(f"  标准差: {ent_array.std():.4f}")
    print(f"  最小值: {ent_array.min():.4f} (step {ent_array.argmin() + 1})")
    print(f"  最大值: {ent_array.max():.4f} (step {ent_array.argmax() + 1})")
    print(f"\nToken Gate Sigmoid 统计:")
    print(f"  平均值: {gate_array.mean():.6f}")
    print(f"  最小值: {gate_array.min():.6f} (step {gate_array.argmin() + 1})")
    print(f"  最大值: {gate_array.max():.6f} (step {gate_array.argmax() + 1})")

    # 打印熵最高的 Top-20 步骤
    top_k = min(20, num_steps)
    top_indices = np.argsort(ent_array)[::-1][:top_k]
    print(f"\n熵最高的 Top-{top_k} 步骤:")
    print("-" * 80)
    print(f"{'Rank':>4}  {'Step':>5}  {'Entropy':>10}  {'GateSigm':>10}  Token")
    print("-" * 80)
    for rank, idx in enumerate(top_indices):
        token_repr = repr(tokens_text[idx])
        print(f"{rank + 1:>4}  {idx + 1:>5}  {entropies[idx]:>10.4f}  {gate_values[idx]:>10.6f}  {token_repr}")
    print("-" * 80)

    # ---- 6. 绘制折线图（双 Y 轴：Entropy + Gate Sigmoid）----
    fig, ax1 = plt.subplots(figsize=(14, 5))
    steps = np.arange(1, num_steps + 1)

    # 左 Y 轴：Entropy
    color_entropy = "steelblue"
    ax1.plot(steps, entropies, linewidth=0.8, color=color_entropy, alpha=0.9, label="Entropy")
    ax1.set_xlabel("Generation Step", fontsize=12)
    ax1.set_ylabel("Entropy (nats)", fontsize=12, color=color_entropy)
    ax1.tick_params(axis="y", labelcolor=color_entropy)

    # 右 Y 轴：Gate Sigmoid Mean
    ax2 = ax1.twinx()
    color_gate = "darkorange"
    ax2.plot(steps, gate_values, linewidth=0.8, color=color_gate, alpha=0.7, label="Gate Sigmoid")
    ax2.set_ylabel("Token Gate Sigmoid Mean", fontsize=12, color=color_gate)
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

    ax1.set_title("Token-level Entropy & Gate Sigmoid during Generation", fontsize=14)
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
        "gate_sigmoid_mean": float(gate_array.mean()),
        "entropies": [float(e) for e in entropies],
        "gate_values": [float(g) for g in gate_values],
        "tokens": tokens_text,
    }
    json_path = os.path.join(save_dir, "entropy_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entropy_data, f, indent=2, ensure_ascii=False)
    print(f"熵数据已保存至: {json_path}")

    return entropies, tokens_text, response_text


if __name__ == "__main__":
    # ====== 在此处手动修改参数，直接运行即可调试 ======
    checkpoint_path = "/root/autodl-tmp/HRPO/test0116.2.0.4/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934"  # 修改为你的 adapter 路径
    temperature = 0.9
    is_inference = True   # True = greedy, False = sampling
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

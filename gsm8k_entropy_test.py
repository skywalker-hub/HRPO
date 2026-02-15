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
QUESTION = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
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

    # ---- 3. 生成回复（与 eval_gsm8k.py 一致的参数）----
    print("\n正在生成回复...")
    with torch.no_grad():
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=temperature,
                max_new_tokens=512,
                output_hidden_states=True,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
        )

    # ---- 4. 解码文本 ----
    generated_ids = outputs[0][prompt_length:]
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

    # ---- 5. 前向传播获取 logits，逐步计算信息熵 ----
    # 对完整序列（prompt + 生成）做一次前向传播，因果注意力保证每个位置
    # 只看到它之前的 token，与自回归生成时完全一致
    full_ids = outputs[0].unsqueeze(0)  # (1, seq_len)
    print("\n正在计算信息熵（前向传播）...")
    with torch.no_grad():
        model_outputs = model(full_ids)
        all_logits = model_outputs.logits  # (1, seq_len, vocab_size)

    # logits[t] 预测 token[t+1]，所以：
    #   all_logits[0, prompt_length-1] 预测第 1 个生成 token
    #   all_logits[0, prompt_length]   预测第 2 个生成 token ...
    num_gen_tokens = len(generated_ids)
    gen_logits = all_logits[0, prompt_length - 1 : prompt_length - 1 + num_gen_tokens, :]

    # 应用 temperature（与生成时一致）
    gen_logits = gen_logits / temperature

    num_steps = gen_logits.size(0)
    entropies = []
    tokens_text = []

    for step_idx in range(num_steps):
        entropy = compute_entropy(gen_logits[step_idx])
        entropies.append(entropy)

        token_id = generated_ids[step_idx].item()
        token_str = tokenizer.decode([token_id])
        tokens_text.append(token_str)

    # 打印前 30 步和最后 10 步的详细信息
    print(f"\n共生成 {num_steps} 个 token")
    print("-" * 50)
    print(f"{'Step':>5}  {'Entropy':>10}  Token")
    print("-" * 50)
    display_steps = list(range(min(30, num_steps)))
    if num_steps > 40:
        display_steps.append(None)  # 省略标记
        display_steps.extend(range(num_steps - 10, num_steps))
    elif num_steps > 30:
        display_steps.extend(range(30, num_steps))

    for step_idx in display_steps:
        if step_idx is None:
            print(f"  ...   {'...':>10}  ...")
        else:
            token_repr = repr(tokens_text[step_idx])
            print(f"{step_idx + 1:>5}  {entropies[step_idx]:>10.4f}  {token_repr}")
    print("-" * 50)

    # 统计摘要
    ent_array = np.array(entropies)
    print(f"\n信息熵统计:")
    print(f"  平均值: {ent_array.mean():.4f}")
    print(f"  标准差: {ent_array.std():.4f}")
    print(f"  最小值: {ent_array.min():.4f} (step {ent_array.argmin() + 1})")
    print(f"  最大值: {ent_array.max():.4f} (step {ent_array.argmax() + 1})")

    # ---- 6. 绘制折线图 ----
    fig, ax = plt.subplots(figsize=(14, 5))
    steps = np.arange(1, num_steps + 1)
    ax.plot(steps, entropies, linewidth=0.8, color="steelblue", alpha=0.9)

    # 标注 #### 答案标记位置
    answer_marker = ANSWER_START
    full_gen_text = ""
    answer_step = None
    for idx, t in enumerate(tokens_text):
        full_gen_text += t
        if answer_marker in full_gen_text and answer_step is None:
            answer_step = idx + 1  # 1-indexed

    if answer_step is not None:
        ax.axvline(x=answer_step, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            answer_step, ax.get_ylim()[1] * 0.95,
            f" {answer_marker}",
            color="red", fontsize=9, va="top",
        )

    ax.set_xlabel("Generation Step", fontsize=12)
    ax.set_ylabel("Entropy (nats)", fontsize=12)
    ax.set_title("Token-level Entropy during Generation", fontsize=14)
    ax.grid(True, alpha=0.3)
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
        "entropies": [float(e) for e in entropies],
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
    temperature = 0.5
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

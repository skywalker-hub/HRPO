"""
ltt_hrpo.py — 在 LT_Tuning 微调模型上进行 HRPO 强化学习训练

流程:
  1. 从 LT_Tuning SFT 阶段产出的模型 (如 models/example) 加载微调后的 LLaMA
  2. 通过 unsloth 的 FastLanguageModel 加载并应用 LoRA
  3. 初始化 HRPO 特有模块 (residual gates, Lambda, head, gate matrix)
  4. 使用 GRPO (Group Relative Policy Optimization) 在 GSM8K 上训练

用法:
  python ltt_hrpo.py --model_path models/example
  python ltt_hrpo.py --model_path models/example --lora_rank 32 --lr 5e-6
  python ltt_hrpo.py --config_file configs/example_config.yaml
"""

import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_rl_dir = os.path.join(_current_dir, "rl_example")
if _rl_dir not in sys.path:
    sys.path.insert(0, _rl_dir)

import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import argparse
import yaml
import torch
import torch.nn as nn
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import (
    ANSWER_START,
    SYSTEM_PROMPT,
    get_reward_func,
    process_gsm8k,
    process_gsm8k_answer,
    extract_hash_answer,
)

os.environ["WANDB_PROJECT"] = "ltt-hrpo"


def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main")[split]
    return dataset.map(
        process_gsm8k, batched=True,
        batch_size=chunk_size, load_from_cache_file=False,
    )


def resolve_model_path(args) -> str:
    """从命令行参数或 YAML 配置文件中解析微调模型路径"""
    if args.model_path:
        return args.model_path

    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            cfg = yaml.safe_load(f)
        save_path = cfg.get("save_path", "models")
        name = cfg.get("name", "example")
        resolved = os.path.join(save_path, name)
        print(f"[ltt_hrpo] 从配置文件 {args.config_file} 解析模型路径: {resolved}")
        return resolved

    raise ValueError(
        "必须通过 --model_path 指定微调模型路径，"
        "或通过 --config_file 指定包含 save_path/name 的 YAML 配置"
    )


def print_module_init_info(head_weight, gate_weight, token_gate_init):
    """打印 HRPO 模块的初始化信息"""
    print("\n" + "=" * 60)
    print("HRPO 新增模块初始值检查")
    print("=" * 60)

    print(f"\n[thinking_residual_head]")
    print(f"  形状: {head_weight.shape}")
    print(f"  最小值: {head_weight.min().item():.6f}")
    print(f"  最大值: {head_weight.max().item():.6f}")
    print(f"  均值: {head_weight.mean().item():.6f}")
    print(f"  是否全为0: {(head_weight == 0).all().item()}")

    print(f"\n[token_gate_matrix]")
    print(f"  形状: {gate_weight.shape}")
    print(f"  最小值: {gate_weight.min().item():.6f}")
    print(f"  最大值: {gate_weight.max().item():.6f}")
    print(f"  均值: {gate_weight.mean().item():.6f}")
    expected = torch.full_like(gate_weight, float(token_gate_init))
    print(f"  是否全为{token_gate_init:g}: {torch.allclose(gate_weight, expected)}")
    sigmoid_vals = torch.sigmoid(gate_weight)
    print(f"  sigmoid后的值范围: [{sigmoid_vals.min().item():.6f}, {sigmoid_vals.max().item():.6f}]")

    print("=" * 60 + "\n")


def print_optimizer_debug(model, trainer):
    """打印优化器参数组信息，用于调试"""
    print("\n" + "=" * 60)
    print("调试：检查参数是否在优化器中")
    print("=" * 60)

    print("\n【所有包含 'token_gate' 的参数】")
    found_gate = False
    for name, param in model.named_parameters():
        if "token_gate" in name:
            found_gate = True
            print(f"  {name}")
            print(f"    shape: {param.shape}, requires_grad: {param.requires_grad}, dtype: {param.dtype}")
    if not found_gate:
        print("  ⚠️ 没有找到任何包含 'token_gate' 的参数！")

    print("\n【优化器参数组】")
    trainer.create_optimizer()
    for i, group in enumerate(trainer.optimizer.param_groups):
        param_count = len(group["params"])
        total_params = sum(p.numel() for p in group["params"])
        print(f"  Group {i}: lr={group['lr']:.2e}, params={param_count}, total={total_params:,}")

        for p in group["params"]:
            for name, param in model.named_parameters():
                if param is p and "token_gate" in name:
                    print(f"    ✓ 包含 token_gate_matrix (lr={group['lr']:.2e})")

    print("=" * 60 + "\n")


def main(args):
    model_path = resolve_model_path(args)
    print(f"\n{'=' * 60}")
    print(f"LTT-HRPO: 加载微调模型并进行强化学习")
    print(f"模型路径: {model_path}")
    print(f"{'=' * 60}\n")

    # ============ 第1步：加载微调后的模型 ============
    max_seq_length = args.max_prompt_length + args.max_completion_length
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        fast_inference=False,
    )
    model.answer_start = ANSWER_START

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    print(f"模型加载完成: vocab_size={len(tokenizer)}")

    # 检查 <thinking> token 是否存在于词表中 (SFT 阶段应已添加)
    thinking_token = "<thinking>"
    if thinking_token in tokenizer.get_vocab():
        thinking_token_id = tokenizer.convert_tokens_to_ids(thinking_token)
        print(f"检测到 <thinking> token (id={thinking_token_id})，SFT 阶段已正确添加")
    else:
        print(f"警告: 词表中未找到 <thinking> token，可能未经过 LT_Tuning SFT 训练")

    # ============ 第2步：应用 LoRA ============
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=[
            "thinking_residual_gate_r",
            "thinking_residual_gate_i",
            "thinking_residual_Lambda",
            "thinking_residual_head",
            "token_gate_matrix",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ============ 第3步：初始化 HRPO 模块 ============
    model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
        r_min=args.residual_r_min, r_max=args.residual_r_max,
    )

    # 获取真正可训练的权重 (PEFT 包装后是 modules_to_save.default.weight)
    head_module = model.model.model.thinking_residual_head
    gate_module = model.model.model.token_gate_matrix

    if hasattr(head_module, "modules_to_save"):
        head_trainable_weight = head_module.modules_to_save.default.weight
    else:
        head_trainable_weight = head_module.weight

    if hasattr(gate_module, "modules_to_save"):
        gate_trainable_weight = gate_module.modules_to_save.default.weight
    else:
        gate_trainable_weight = gate_module.weight

    token_gate_init = args.token_gate_init
    nn.init.zeros_(head_trainable_weight)
    nn.init.constant_(gate_trainable_weight, token_gate_init)

    print(f"\n初始化完成:")
    print(f"  thinking_residual_head: 使用 {'modules_to_save.default' if hasattr(head_module, 'modules_to_save') else 'weight'}")
    print(f"  token_gate_matrix: 使用 {'modules_to_save.default' if hasattr(gate_module, 'modules_to_save') else 'weight'}")

    print_module_init_info(head_trainable_weight.data, gate_trainable_weight.data, token_gate_init)

    # ============ 第4步：构建实验名称与输出路径 ============
    model_short_name = os.path.basename(model_path.rstrip("/"))
    exp_name = (
        f"./ltt_hrpo_outputs/{model_short_name}-gsm8k"
        f"-group{args.group_size}-lora{args.lora_rank}"
        f"-lr{args.lr_token_gate_matrix}-init{token_gate_init:g}"
        f"-rmin{args.residual_r_min}-temp{args.temperature}"
    )
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"实验目录 {exp_name} 已存在且非空。退出...")
        return

    print(f"输出路径: {exp_name}")

    # ============ 第5步：配置 GRPO 训练 ============
    training_args = GRPOConfig(
        use_vllm=False,
        learning_rate=args.lr,
        beta=args.beta,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        temperature=args.temperature,
        num_generations=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        output_dir=exp_name,
    )

    # ============ 第6步：准备数据集与训练器 ============
    dataset = preprocess_gsm8k("train", chunk_size=500)
    print(f"GSM8K 训练集加载完成: {len(dataset)} 样本")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            get_reward_func(process_gsm8k_answer),
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # 为不同模块设置差异化学习率
    patch_trainer_optimizer(
        trainer,
        args.lr_residual_gate,
        args.lr_residual_Lambda,
        args.lr_residual_head,
        args.lr_token_gate_matrix,
    )

    print_optimizer_debug(model, trainer)

    # ============ 第7步：启动训练 ============
    print(f"\n{'=' * 60}")
    print(f"开始 HRPO 训练")
    print(f"  模型: {model_path}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  学习率: {args.lr}")
    print(f"  Group size: {args.group_size}")
    print(f"  温度: {args.temperature}")
    print(f"  门控初始值: {token_gate_init:g} (sigmoid={torch.sigmoid(torch.tensor(token_gate_init)).item():.4f})")
    print(f"{'=' * 60}\n")

    trainer.train()
    print("训练完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LTT-HRPO: 在 LT_Tuning 微调模型上进行 HRPO 强化学习"
    )

    # 模型路径
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="微调后的模型路径 (如 models/example)",
    )
    parser.add_argument(
        "--config_file", type=str, default=None,
        help="LT_Tuning 的 YAML 配置文件路径，用于自动解析模型路径",
    )

    # LoRA 配置
    parser.add_argument("--lora_rank", type=int, default=32)

    # 学习率
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--lr_residual_gate", type=float, default=1e-4)
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)
    parser.add_argument("--lr_residual_head", type=float, default=1e-4)
    parser.add_argument("--lr_token_gate_matrix", type=float, default=1e-2)

    # HRPO 模块初始化
    parser.add_argument("--residual_r_min", type=float, default=0.981)
    parser.add_argument("--residual_r_max", type=float, default=0.999)
    parser.add_argument("--token_gate_init", type=float, default=-2.0)

    # 优化器
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    # GRPO 配置
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)

    # 训练控制
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
import argparse
import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import *

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    return dataset.map(process_gsm8k, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_prompt_length + args.max_completion_length,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save = [
            "thinking_residual_decay",
            "thinking_residual_head",
            "token_gate_matrix",
        ], 
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )

    # ============ 【真正生效的初始化】 ============
    # post_init() 会覆盖模型定义中的初始化，PEFT 包装后需要重新初始化 modules_to_save.default
    import torch.nn as nn

    # --- thinking_residual_decay: Lambda 初始化 + proj 零初始化 ---
    decay_module = model.model.model.thinking_residual_decay
    if hasattr(decay_module, 'modules_to_save'):
        decay_inner = decay_module.modules_to_save.default
    else:
        decay_inner = decay_module
    # Lambda: 通过 sigmoid 反函数映射到目标初值范围
    with torch.no_grad():
        nn.init.uniform_(decay_inner.Lambda, a=args.residual_r_min, b=args.residual_r_max)
        decay_inner.Lambda.data.copy_(torch.log(decay_inner.Lambda / (1.0 - decay_inner.Lambda)))
    # proj: 零初始化，保证训练初期 a_t 仅由 Lambda 决定
    nn.init.zeros_(decay_inner.proj.weight)

    # --- thinking_residual_head: 零初始化 ---
    head_module = model.model.model.thinking_residual_head
    if hasattr(head_module, 'modules_to_save'):
        head_trainable_weight = head_module.modules_to_save.default.weight
    else:
        head_trainable_weight = head_module.weight
    nn.init.zeros_(head_trainable_weight)

    # --- token_gate_matrix: 常数初始化 ---
    gate_module = model.model.model.token_gate_matrix
    if hasattr(gate_module, 'modules_to_save'):
        gate_trainable_weight = gate_module.modules_to_save.default.weight
    else:
        gate_trainable_weight = gate_module.weight
    token_gate_init = -2.0
    nn.init.constant_(gate_trainable_weight, token_gate_init)

    # --- 保存文件名 ---
    exp_name = (f"./main319.2.0/{args.model_name.split('/')[-1]}-gsm8k-group{args.group_size}"
                f"-lora{args.lora_rank}-lr{args.lr_token_gate_matrix}-init{token_gate_init:g}"
                f"-rmin{args.residual_r_min}-temp{args.temperature}")
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

    # ============ 打印初始值检查 ============
    print("\n" + "=" * 60)
    print("HRPO 模块初始值检查")
    print("=" * 60)

    lambda_data = decay_inner.Lambda.data
    sigmoid_lambda = torch.sigmoid(lambda_data)
    print(f"\n[thinking_residual_decay.Lambda]")
    print(f"  形状: {lambda_data.shape}")
    print(f"  原始值范围: [{lambda_data.min().item():.4f}, {lambda_data.max().item():.4f}]")
    print(f"  sigmoid(Lambda) 范围: [{sigmoid_lambda.min().item():.6f}, {sigmoid_lambda.max().item():.6f}]")
    print(f"  (期望初始 a_t 在 [{args.residual_r_min}, {args.residual_r_max}])")

    proj_weight = decay_inner.proj.weight.data
    print(f"\n[thinking_residual_decay.proj]")
    print(f"  形状: {proj_weight.shape}")
    print(f"  是否全为0: {(proj_weight == 0).all().item()}")

    print(f"\n[thinking_residual_head]")
    print(f"  形状: {head_trainable_weight.data.shape}")
    print(f"  是否全为0: {(head_trainable_weight.data == 0).all().item()}")

    gate_weight = gate_trainable_weight.data
    print(f"\n[token_gate_matrix]")
    print(f"  形状: {gate_weight.shape}")
    print(f"  均值: {gate_weight.mean().item():.6f}")
    print(f"  sigmoid后的值范围: [{torch.sigmoid(gate_weight).min().item():.6f}, {torch.sigmoid(gate_weight).max().item():.6f}]")

    print("=" * 60 + "\n")

    training_args = GRPOConfig(
        use_vllm = False,
        learning_rate = args.lr,
        beta = args.beta,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optimizer,
        max_grad_norm = args.max_grad_norm,
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        temperature = args.temperature,
        num_generations = args.group_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = 1,
        save_steps = 250,
        save_total_limit = 3,
        report_to = "wandb",
        output_dir = exp_name,
    )

    dataset = preprocess_gsm8k('train', chunk_size=500)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_reward_func(process_gsm8k_answer),
        ],
        args = training_args,
        train_dataset = dataset,
    )
    patch_trainer_optimizer(
        trainer,
        args.lr_residual_decay,
        args.lr_residual_decay,  # 兼容旧参数位置
        args.lr_residual_head,
        args.lr_token_gate_matrix,
    )
    
    # ============ 调试：检查 token_gate_matrix 是否被正确加入优化器 ============
    print("\n" + "=" * 60)
    print("调试：检查参数是否在优化器中")
    print("=" * 60)
    
    # 检查所有参数名
    print("\n【所有包含 'token_gate' 的参数】")
    found_gate = False
    for name, param in model.named_parameters():
        if "token_gate" in name:
            found_gate = True
            print(f"  {name}")
            print(f"    shape: {param.shape}, requires_grad: {param.requires_grad}, dtype: {param.dtype}")
    if not found_gate:
        print("  ⚠️ 没有找到任何包含 'token_gate' 的参数！")
    
    # 检查优化器中的参数组
    print("\n【优化器参数组】")
    trainer.create_optimizer()
    for i, group in enumerate(trainer.optimizer.param_groups):
        param_count = len(group['params'])
        total_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: lr={group['lr']:.2e}, params={param_count}, total={total_params:,}")
        
        # 检查是否有 token_gate_matrix 参数
        for p in group['params']:
            for name, param in model.named_parameters():
                if param is p and "token_gate" in name:
                    print(f"    ✓ 包含 token_gate_matrix (lr={group['lr']:.2e})")
    
    print("=" * 60 + "\n")
    # ============ 调试结束 ============
    
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--residual_r_min", type=float, default=0.981)
    parser.add_argument("--residual_r_max", type=float, default=0.999)
    parser.add_argument("--lr_residual_decay", type=float, default=1e-4)

    # 新增: 隐状态变换头的学习率
    parser.add_argument("--lr_residual_head", type=float, default=1e-4)  
    # 新增: Token 门控矩阵的学习率 (提高以克服bfloat16精度问题)
    parser.add_argument("--lr_token_gate_matrix", type=float, default=1e-2)  
    
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)

    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"

    main(args)

    ###日志：本代码仅对HRPO的h做了替换，加入了一个线性头训练。
    ###1-21日志：本代码再次尝试加入门控矩阵
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
            # "thinking_residual_gate_r",
            # "thinking_residual_gate_i",
            # "thinking_residual_Lambda",
            # "thinking_residual_head",
            # "token_gate_matrix",
            "z_q_proj",  # 连续路径 Q 投影矩阵
            "z_k_proj",  # 连续路径 K 投影矩阵
            "z_v_proj",  # 连续路径 V 投影矩阵
        ], 
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )
    # model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
    #     r_min = args.residual_r_min, r_max = args.residual_r_max,
    # )
    
    # ============ 【真正生效的初始化】 ============
    # 注意：模型定义中的初始化会被 post_init() 覆盖，PEFT 包装后需要初始化 modules_to_save.default
    # 这里才是真正决定训练初始值的地方！
    import torch.nn as nn
    
    # # -------- 旧方案: thinking_residual 相关初始化 (已注释) --------
    # # 获取真正可训练的权重（PEFT 包装后是 modules_to_save.default.weight）
    # head_module = model.model.model.thinking_residual_head
    # gate_module = model.model.model.token_gate_matrix
    # 
    # if hasattr(head_module, 'modules_to_save'):
    #     head_trainable_weight = head_module.modules_to_save.default.weight
    # else:
    #     head_trainable_weight = head_module.weight
    # 
    # if hasattr(gate_module, 'modules_to_save'):
    #     gate_trainable_weight = gate_module.modules_to_save.default.weight
    # else:
    #     gate_trainable_weight = gate_module.weight
    # 
    # nn.init.zeros_(head_trainable_weight)  # thinking_residual_head: 初始化为 0
    # token_gate_init = -2.0
    # nn.init.constant_(gate_trainable_weight, token_gate_init)
    # -------- 旧方案结束 --------

    # -------- 新方案: z_*_proj 零初始化 --------
    z_proj_names = ["z_q_proj", "z_k_proj", "z_v_proj"]
    for proj_name in z_proj_names:
        proj_module = getattr(model.model.model, proj_name)
        if hasattr(proj_module, 'modules_to_save'):
            nn.init.zeros_(proj_module.modules_to_save.default.weight)
        else:
            nn.init.zeros_(proj_module.weight)

    exp_name = (f"./test301.2/{args.model_name.split('/')[-1]}-gsm8k-group{args.group_size}"
                f"-lora{args.lora_rank}-lr{args.lr_z_proj}"
                f"-temp{args.temperature}")
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

    # ============ 打印初始值情况 ============
    print("\n" + "=" * 60)
    print("连续路径 QKV 投影矩阵初始值检查")
    print("=" * 60)
    for proj_name in z_proj_names:
        proj_module = getattr(model.model.model, proj_name)
        w = proj_module.modules_to_save.default.weight.data if hasattr(proj_module, 'modules_to_save') else proj_module.weight.data
        print(f"\n[{proj_name}]")
        print(f"  形状: {w.shape}")
        print(f"  是否全为0: {(w == 0).all().item()}")
        print(f"  requires_grad: {w.requires_grad}")
    print("=" * 60 + "\n")
    # ============ 初始值检查结束 ============

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
        args.lr_residual_gate,
        args.lr_residual_Lambda,
        args.lr_residual_head,
        args.lr_token_gate_matrix,
        lr_z_proj = args.lr_z_proj,  # 连续路径 QKV 投影矩阵的学习率
    )
    
    # ============ 调试：检查 z_*_proj 是否被正确加入优化器 ============
    print("\n" + "=" * 60)
    print("调试：检查参数是否在优化器中")
    print("=" * 60)

    print("\n【所有包含 'z_q_proj/z_k_proj/z_v_proj' 的参数】")
    found_z = False
    for name, param in model.named_parameters():
        if any(z in name for z in ("z_q_proj", "z_k_proj", "z_v_proj")):
            found_z = True
            print(f"  {name}")
            print(f"    shape: {param.shape}, requires_grad: {param.requires_grad}, dtype: {param.dtype}")
    if not found_z:
        print("  WARNING: 没有找到任何 z_*_proj 参数！")

    print("\n【优化器参数组】")
    trainer.create_optimizer()
    for i, group in enumerate(trainer.optimizer.param_groups):
        param_count = len(group['params'])
        total_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: lr={group['lr']:.2e}, params={param_count}, total={total_params:,}")

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
    parser.add_argument("--lr_residual_gate", type=float, default=1e-4)
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)

    parser.add_argument("--lr_residual_head", type=float, default=1e-4)
    parser.add_argument("--lr_token_gate_matrix", type=float, default=1e-2)
    # 连续路径 QKV 投影矩阵的学习率
    parser.add_argument("--lr_z_proj", type=float, default=1e-4)
    
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
    ###2-23日止：本次尝试和熵结合
    ###3-1日志：test0121方法
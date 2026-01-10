import os
os.environ['UNSLOTH_DISABLE_AUTO_UPDATES'] = '1' 

import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
import argparse
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer, ThinkingModulesMonitorCallback
from utils import *

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    return dataset.map(process_gsm8k, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-gsm8k-group{args.group_size}"
                f"-lora{args.lora_rank}-temp{args.temperature}-110")
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

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
            "info_head",
            "token_gate_matrix",
        ], 
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )

    # ============================================================
    # 验证 info_head 和 token_gate_matrix 的初始化值
    # ============================================================
    print("\n" + "=" * 60)
    print("检查 Token-Dependent Gated Residual 模块初始化")
    print("=" * 60)
    
    # 获取 PEFT 包装后的模块
    info_head_wrapper = model.model.model.info_head
    token_gate_wrapper = model.model.model.token_gate_matrix
    
    # 检查 info_head (预期: mean≈0, std≈0.001)
    if hasattr(info_head_wrapper, 'modules_to_save'):
        ih_weight = info_head_wrapper.modules_to_save.default.weight
        print(f"\ninfo_head (modules_to_save.default):")
        print(f"  Shape: {ih_weight.shape}")
        print(f"  Mean:  {ih_weight.mean().item():.6f} (预期: ≈0)")
        print(f"  Std:   {ih_weight.std().item():.6f} (预期: ≈0.001)")
    else:
        ih_weight = info_head_wrapper.weight
        print(f"\ninfo_head (未被PEFT包装!):")
        print(f"  Mean: {ih_weight.mean().item():.6f}, Std: {ih_weight.std().item():.6f}")
    
    # 检查 token_gate_matrix (预期: mean=-4.0, std=0)
    if hasattr(token_gate_wrapper, 'modules_to_save'):
        tg_weight = token_gate_wrapper.modules_to_save.default.weight
        print(f"\ntoken_gate_matrix (modules_to_save.default):")
        print(f"  Shape: {tg_weight.shape}")
        print(f"  Mean:  {tg_weight.mean().item():.6f} (预期: -4.0)")
        print(f"  Std:   {tg_weight.std().item():.6f} (预期: ≈0)")
        print(f"  Min:   {tg_weight.min().item():.6f}")
        print(f"  Max:   {tg_weight.max().item():.6f}")
        
        # 警告检查
        if abs(tg_weight.mean().item() - (-4.0)) > 0.1:
            print(f"\n  ⚠️  警告: token_gate_matrix 未正确初始化为 -4.0!")
            print(f"      这将导致门一开始就半开 (sigmoid(0)=0.5) 而不是几乎关闭 (sigmoid(-4)≈0.018)")
        else:
            print(f"\n  ✅ token_gate_matrix 正确初始化为 -4.0")
    else:
        tg_weight = token_gate_wrapper.weight
        print(f"\ntoken_gate_matrix (未被PEFT包装!):")
        print(f"  Mean: {tg_weight.mean().item():.6f}, Std: {tg_weight.std().item():.6f}")
    
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
    
    # 创建监控回调
    thinking_monitor = ThinkingModulesMonitorCallback()
    
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_reward_func(process_gsm8k_answer),
        ],
        args = training_args,
        train_dataset = dataset,
        callbacks = [thinking_monitor],  # 添加监控回调
    )
    
    # 设置 trainer 引用给回调（用于存储梯度范数）
    thinking_monitor.set_trainer(trainer)
    
    patch_trainer_optimizer(
        trainer,
        args.lr_info_head,
        args.lr_token_gate_matrix,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--lr_info_head", type=float, default=1e-4)
    parser.add_argument("--lr_token_gate_matrix", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    
    #parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    #parser.add_argument("--per_device_train_batch_size", type=int, default=4)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)

    
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
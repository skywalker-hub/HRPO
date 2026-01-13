"""
用于手动调试 Qwen 推理流程的测试脚本
不使用命令行参数，所有配置硬编码，方便在 IDE 中打断点调试

调试建议的断点位置:
1. model.generate() 调用处 - 进入生成流程
2. transformers/generation/utils.py 的 _sample() 函数 - 采样循环
3. unsloth/models/llama.py 的 CausalLM_fast_forward() - 前向传播入口
4. unsloth/models/llama.py 的 LlamaModel_fast_forward() - 模型主体前向
5. unsloth/models/llama.py 的 LlamaAttention_fast_forward() - 注意力计算
"""

import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig

from utils import *


# ============================================
# 配置区域 - 在这里修改你的调试参数
# ============================================

# 基础模型路径 (根据你的环境修改)
BASE_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
# 或者使用 HuggingFace 路径: "Qwen/Qwen2.5-1.5B-Instruct"

# Adapter 路径 (你训练好的 LoRA adapter)
ADAPTER_PATH = "/root/autodl-tmp/checkpoints/your_adapter_path"

# 生成参数
TEMPERATURE = 0.9
IS_INFERENCE = True  # True = greedy (argmax), False = sampling

# 调试参数 - 小批量便于调试
BATCH_SIZE = 1       # 建议调试时用 1
NUM_SAMPLES = 2      # 只测试 2 个样本，方便快速调试
SAVE_RESULTS = False # 调试时不保存结果

# ============================================


def load_model_and_tokenizer():
    """加载模型和分词器 - 可以在这里打断点观察模型加载过程"""
    print(f"Loading model from: {BASE_MODEL_PATH}")
    print(f"Loading adapter from: {ADAPTER_PATH}")
    
    # 断点1: 进入 FastLanguageModel.from_pretrained 查看模型加载
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=False,
        fast_inference=False,
    )
    
    # 设置 answer_start 用于 HRPO 的 thinking 逻辑
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载 adapter
    model.load_adapter(ADAPTER_PATH)
    
    # 断点2: 进入 for_inference 查看推理模式设置
    model = FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def prepare_single_sample(tokenizer, question: str):
    """准备单个样本的输入 - 方便观察 tokenization 过程"""
    # 构建对话格式
    prompt = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question.strip()},
    ]
    
    # 应用 chat template
    formatted_prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("=" * 50)
    print("Formatted prompt:")
    print(formatted_prompt)
    print("=" * 50)
    
    # Tokenize
    prompt_inputs = tokenizer(
        [formatted_prompt],  # 包装成 list
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False
    )
    
    return prompt_inputs, formatted_prompt


def generate_response(model, tokenizer, prompt_inputs):
    """
    生成响应 - 核心调试位置
    
    在这里打断点，然后 step into model.generate() 可以追踪:
    1. transformers/generation/utils.py :: generate()
    2. transformers/generation/utils.py :: _sample()
    3. unsloth/models/llama.py :: CausalLM_fast_forward()
    4. unsloth/models/llama.py :: LlamaModel_fast_forward()
    """
    prompt_ids = prompt_inputs["input_ids"].to(model.device)
    prompt_mask = prompt_inputs["attention_mask"].to(model.device)
    prompt_length = prompt_ids.size(1)
    
    print(f"Input shape: {prompt_ids.shape}")
    print(f"Prompt length: {prompt_length}")
    
    # ★★★ 核心断点位置 ★★★
    # Step into 这个 generate 调用来追踪整个推理流程
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        generation_config=GenerationConfig(
            do_sample=True,        # 启用采样 (temperature 生效)
            temperature=TEMPERATURE,
            max_new_tokens=512,
            output_hidden_states=True,  # 输出隐藏状态 (HRPO 需要)
        ),
        processing_class=tokenizer,
        is_inference=IS_INFERENCE,  # True=argmax, False=multinomial
    )
    
    return outputs, prompt_length


def decode_and_evaluate(tokenizer, outputs, prompt_length, true_answer_str):
    """解码输出并评估"""
    output = outputs[0]  # 取第一个样本
    
    # 解码生成的 tokens
    response = tokenizer.decode(output[prompt_length:])
    response = response.split(tokenizer.special_tokens_map['eos_token'])[0]
    
    print("=" * 50)
    print("Generated response:")
    print(response)
    print("=" * 50)
    
    # 提取答案
    extracted = extract_from_response(response)
    generated_answer = process_gsm8k_answer(extracted)
    true_answer = extract_hash_answer(true_answer_str)
    true_answer = process_gsm8k_answer(true_answer)
    
    is_correct = generated_answer == true_answer
    
    print(f"Generated answer: {generated_answer}")
    print(f"True answer: {true_answer}")
    print(f"Correct: {is_correct}")
    
    return {
        'response': response,
        'generated_answer': generated_answer,
        'true_answer': true_answer,
        'correct': is_correct
    }


def main():
    """主函数 - 调试入口"""
    print("=" * 60)
    print("GSM8K Debug Test Script")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Inference mode (greedy): {IS_INFERENCE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num samples: {NUM_SAMPLES}")
    print("=" * 60)
    
    # Step 1: 加载模型
    print("\n[Step 1] Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model device: {model.device}")
    print(f"Model type: {type(model)}")
    
    # Step 2: 加载数据集
    print("\n[Step 2] Loading dataset...")
    dataset = load_dataset('openai/gsm8k', 'main')['test']
    if NUM_SAMPLES and len(dataset) > NUM_SAMPLES:
        dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    print(f"Loaded {len(dataset)} samples")
    
    # Step 3: 逐个处理样本 (便于调试)
    results = []
    correct = 0
    
    for i, sample in enumerate(dataset):
        print(f"\n{'=' * 60}")
        print(f"[Sample {i + 1}/{len(dataset)}]")
        print(f"Question: {sample['question'][:100]}...")
        print("=" * 60)
        
        # 准备输入
        prompt_inputs, formatted_prompt = prepare_single_sample(
            tokenizer, sample['question']
        )
        
        # ★ 生成响应 - 主要调试位置 ★
        outputs, prompt_length = generate_response(
            model, tokenizer, prompt_inputs
        )
        
        # 解码和评估
        result = decode_and_evaluate(
            tokenizer, outputs, prompt_length, sample['answer']
        )
        result['question'] = sample['question']
        results.append(result)
        
        if result['correct']:
            correct += 1
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / len(results) * 100:.2f}%")
    
    # 可选: 保存结果
    if SAVE_RESULTS and ADAPTER_PATH:
        save_path = ADAPTER_PATH + "/debug_eval_results.json"
        with open(save_path, 'w') as f:
            json.dump({
                'metrics': {
                    'accuracy': correct / len(results),
                    'correct': correct,
                    'total': len(results),
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {save_path}")
    
    return results


if __name__ == "__main__":
    # 直接运行，无需命令行参数
    # 在 IDE 中可以直接 F5 或点击运行按钮
    results = main()

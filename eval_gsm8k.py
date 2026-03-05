import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm

from utils import *


def evaluate_model(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    n_generations: int = 1,
    pass_k_list: list = None,
):
    if pass_k_list is None:
        pass_k_list = [1]
    for k in pass_k_list:
        if k > n_generations:
            raise ValueError(f"pass@{k} requires n_generations >= {k}, got {n_generations}")

    if n_generations > 1 and is_inference:
        print("WARNING: n_generations > 1 时贪心解码会产生相同结果，自动切换为采样模式")
        is_inference = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 1024,
        load_in_4bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    dataset = load_dataset('openai/gsm8k', 'main')['test']
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples, n_generations={n_generations}, pass@k={pass_k_list}")

    all_question_results = []
    total_gen_tokens = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="questions",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'pass@1': '0.00%'})

    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data['question'])

        formatted_prompts = []
        for q in batch_data['question']:
            prompt = tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': q.strip()},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _ in range(n_generations):
                formatted_prompts.append(prompt)

        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask, 
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=temperature,
                max_new_tokens=512,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
        )

        for output in outputs:
            total_gen_tokens += output[prompt_length:].shape[0]

        for q_idx in range(current_batch_size):
            true_answer = extract_hash_answer(batch_data['answer'][q_idx])
            true_answer = process_gsm8k_answer(true_answer)

            generations = []
            for gen_idx in range(n_generations):
                flat_idx = q_idx * n_generations + gen_idx
                output = outputs[flat_idx]
                response = tokenizer.decode(output[prompt_length:])
                response = response.split(tokenizer.special_tokens_map['eos_token'])[0]

                extracted = extract_from_response(response)
                generated_answer = process_gsm8k_answer(extracted)

                generations.append({
                    'generated_answer': generated_answer,
                    'full_response': response,
                    'correct': generated_answer == true_answer,
                })

            all_question_results.append({
                'question': batch_data['question'][q_idx],
                'true_answer': true_answer,
                'generations': generations,
            })

        n_questions = len(all_question_results)
        avg_gen_len = total_gen_tokens / (n_questions * n_generations) if n_questions > 0 else 0
        postfix = {'avg_len': f'{avg_gen_len:.1f}'}
        for k in pass_k_list:
            passed = sum(1 for qr in all_question_results if any(g['correct'] for g in qr['generations'][:k]))
            postfix[f'pass@{k}'] = f'{passed/n_questions*100:.2f}%'
        progress_bar.update(current_batch_size)
        progress_bar.set_postfix(postfix)

    progress_bar.close()

    n_questions = len(all_question_results)
    avg_gen_len = total_gen_tokens / (n_questions * n_generations) if n_questions > 0 else 0
    print(f"\nFinal average generation length: {avg_gen_len:.1f} tokens")

    metrics = {
        'n_generations': n_generations,
        'total_questions': n_questions,
        'avg_gen_length': avg_gen_len,
        'model_path': adapter_path,
        'timestamp': datetime.now().isoformat(),
    }
    for k in pass_k_list:
        passed = sum(1 for qr in all_question_results if any(g['correct'] for g in qr['generations'][:k]))
        metrics[f'pass@{k}'] = passed / n_questions
        print(f"  pass@{k}: {passed}/{n_questions} = {passed/n_questions*100:.2f}%")

    if save_results:
        save_path = adapter_path + "/eval_results.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': all_question_results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--n_generations", type=int, default=32)
    parser.add_argument("--pass_k", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    args = parser.parse_args()

    base_model = None
    checkpoint_path = args.checkpoint_path
    # 本地模型路径映射
    local_model_paths = {
        "Qwen2.5-1.5B-Instruct": "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
        "Llama-3.2-1B-Instruct": "/root/autodl-tmp/models/Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct": "/root/autodl-tmp/models/Llama-3.2-3B-Instruct",
    }
    base_models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    for model in base_models:
        model_name = model.split('/')[-1]
        if model_name in checkpoint_path:
            # 使用本地路径
            base_model = local_model_paths.get(model_name, model)
    temperature = args.temperature
    print(f"checkpoint: {checkpoint_path}\nbase_model: {base_model}\n"
          f"temperature: {temperature}, n={args.n_generations}, pass@k={args.pass_k}")

    if 'eval_results.json' not in os.listdir(checkpoint_path):
        print(f"Starting GSM8k evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=None,
            save_results=True,
            n_generations=args.n_generations,
            pass_k_list=args.pass_k,
        )
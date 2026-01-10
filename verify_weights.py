"""
验证 info_head 和 token_gate_matrix 权重是否被正确保存和训练
"""
import sys
from safetensors import safe_open
import json

def verify_adapter_weights(adapter_path: str):
    """读取并验证 adapter 文件中的权重"""
    
    # 1. 读取 adapter_config.json
    config_path = f"{adapter_path}/adapter_config.json"
    print("=" * 60)
    print("1. 检查 adapter_config.json")
    print("=" * 60)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"modules_to_save: {config.get('modules_to_save', 'NOT FOUND!')}")
        print(f"target_modules: {config.get('target_modules', [])}")
        print(f"r (LoRA rank): {config.get('r', 'N/A')}")
    except Exception as e:
        print(f"Error reading config: {e}")
    
    # 2. 读取 adapter_model.safetensors
    safetensors_path = f"{adapter_path}/adapter_model.safetensors"
    print("\n" + "=" * 60)
    print("2. 检查 adapter_model.safetensors 中的权重")
    print("=" * 60)
    
    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
            
            # 找到 info_head 和 token_gate_matrix 相关的键
            info_head_keys = [k for k in all_keys if 'info_head' in k]
            token_gate_keys = [k for k in all_keys if 'token_gate_matrix' in k]
            lora_keys = [k for k in all_keys if 'lora' in k.lower()]
            
            print(f"\n总共 {len(all_keys)} 个权重张量")
            print(f"- LoRA 相关: {len(lora_keys)} 个")
            print(f"- info_head 相关: {len(info_head_keys)} 个")
            print(f"- token_gate_matrix 相关: {len(token_gate_keys)} 个")
            
            # 打印 info_head 权重信息
            print("\n" + "-" * 60)
            print("info_head 权重详情:")
            print("-" * 60)
            if info_head_keys:
                for key in info_head_keys:
                    tensor = f.get_tensor(key)
                    print(f"\n  Key: {key}")
                    print(f"  Shape: {tensor.shape}")
                    print(f"  Dtype: {tensor.dtype}")
                    print(f"  Min: {tensor.min().item():.6f}")
                    print(f"  Max: {tensor.max().item():.6f}")
                    print(f"  Mean: {tensor.mean().item():.6f}")
                    print(f"  Std: {tensor.std().item():.6f}")
                    
                    # 检查是否为初始值 (N(0, 0.001))
                    if abs(tensor.mean().item()) < 0.01 and tensor.std().item() < 0.01:
                        print(f"  ⚠️  警告: 权重接近初始值 N(0, 0.001)，可能未被充分训练")
                    else:
                        print(f"  ✅ 权重已偏离初始值，说明已被训练")
            else:
                print("  ❌ 未找到 info_head 权重!")
            
            # 打印 token_gate_matrix 权重信息
            print("\n" + "-" * 60)
            print("token_gate_matrix 权重详情:")
            print("-" * 60)
            if token_gate_keys:
                for key in token_gate_keys:
                    tensor = f.get_tensor(key)
                    print(f"\n  Key: {key}")
                    print(f"  Shape: {tensor.shape}")
                    print(f"  Dtype: {tensor.dtype}")
                    print(f"  Min: {tensor.min().item():.6f}")
                    print(f"  Max: {tensor.max().item():.6f}")
                    print(f"  Mean: {tensor.mean().item():.6f}")
                    print(f"  Std: {tensor.std().item():.6f}")
                    
                    # 检查是否为初始值 (-4.0)
                    if abs(tensor.mean().item() - (-4.0)) < 0.1:
                        print(f"  ⚠️  警告: 权重接近初始值 -4.0，可能未被充分训练")
                    else:
                        print(f"  ✅ 权重已偏离初始值 -4.0，说明已被训练")
                        print(f"     (初始值为 -4.0, sigmoid(-4)≈0.018)")
            else:
                print("  ❌ 未找到 token_gate_matrix 权重!")
            
            # 打印部分 LoRA 权重作为对比
            print("\n" + "-" * 60)
            print("LoRA 权重示例 (前3个):")
            print("-" * 60)
            for key in lora_keys[:3]:
                tensor = f.get_tensor(key)
                print(f"\n  Key: {key}")
                print(f"  Shape: {tensor.shape}, Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
            
            # 打印所有键名
            print("\n" + "-" * 60)
            print("所有权重键名:")
            print("-" * 60)
            for key in all_keys:
                print(f"  {key}")
                
    except Exception as e:
        print(f"Error reading safetensors: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_weights.py <adapter_path>")
        print("Example: python verify_weights.py ./experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-temp0.5-tdgr/checkpoint-250")
        sys.exit(1)
    
    adapter_path = sys.argv[1]
    print(f"\n验证 adapter 路径: {adapter_path}\n")
    verify_adapter_weights(adapter_path)

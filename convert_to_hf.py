import os
import argparse
import torch
import json
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 내부 헬퍼 함수들 (동일)
# ==========================================

def _load_state_dict_from_file(file_path):
    print(f"📂 Loading raw weights from: {file_path}")
    state_dict = torch.load(file_path, map_location="cpu")
    if "state_dict" in state_dict: return state_dict["state_dict"]
    elif "model" in state_dict: return state_dict["model"]
    elif "state" in state_dict: return state_dict["state"]
    return state_dict

def _clean_state_dict_keys(state_dict):
    print("🧹 Cleaning and De-quantizing weights (Int8 -> BF16)...")
    new_state_dict = {}
    
    # 1. SCB(Scale) 값들을 먼저 수집 (메모리에 로드)
    scb_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
        if "SCB" in clean_k:
            # SCB 키 저장: 'model...weight' 형태로 매핑하기 위해 변환
            target_weight_key = clean_k.replace(".SCB", ".weight")
            scb_dict[target_weight_key] = v

    # 2. 가중치 순회 및 복구
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "").replace("module.", "")
        
        # 메타데이터 키는 저장 안 함
        if any(bad in new_k for bad in ["SCB", "weight_format", "_scale", "_amax", "_extra_state"]):
            continue

        if torch.is_tensor(v):
            # (A) Int8 텐서인 경우 -> 복구(De-quantization) 수행
            if v.dtype == torch.int8 or v.dtype == torch.uint8:
                # 계산을 위해 float32로 변환
                v = v.to(device="cpu", dtype=torch.float32)
                
                # 짝꿍 SCB가 있는지 확인
                if new_k in scb_dict:
                    scale = scb_dict[new_k].to(device="cpu", dtype=torch.float32)
                    
                    # [공식 적용] Real = (Int8 / 127.0) * SCB
                    # SCB Shape이 (Out_dim,) 이므로 Broadcasting을 위해 reshape 필요
                    if scale.ndim == 1 and v.ndim == 2:
                        if scale.shape[0] == v.shape[0]:
                            scale = scale.view(-1, 1) # (N) -> (N, 1)
                        elif scale.shape[0] == v.shape[1]:
                             # 혹시나 Transpose된 경우
                             scale = scale.view(1, -1)
                    
                    v = (v / 127.0) * scale
                else:
                    # SCB가 없는데 Int8이다? (매우 드문 케이스, 그냥 형변환)
                    print(f"⚠️ Warning: Int8 weight found without SCB: {new_k}")
            
            else:
                # 이미 FP16/BF16/FP32인 경우 (LayerNorm 등) -> 그냥 BF16으로 맞춤
                v = v.to(device="cpu", dtype=torch.bfloat16)

            # 최종 저장 포맷은 bfloat16
            v = v.to(dtype=torch.bfloat16)
            
        new_state_dict[new_k] = v
        
    return new_state_dict

def _save_as_lora(config, state_dict, output_dir):
    # (변환 로직은 .bin으로 저장하도록 유지 - safetensors 라이브러리 의존성 최소화)
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    
    lora_state_dict = {}
    for k, v in state_dict.items():
        if "lora" in k:
            clean_k = k.replace("base_model.model.", "")
            lora_state_dict[clean_k] = v
            
    if not lora_state_dict:
        print("⚠️ Warning: No 'lora' keys found in policy.pt")
        return None

    print(f"   > Saving adapter_model.bin to {adapter_dir}...")
    torch.save(lora_state_dict, os.path.join(adapter_dir, "adapter_model.bin"))
    
    peft_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config.lora.get("r", 8),
        "lora_alpha": config.lora.get("lora_alpha", 16),
        "lora_dropout": config.lora.get("lora_dropout", 0.05),
        "target_modules": list(config.lora.get("target_modules", ["q_proj", "v_proj"])),
        "base_model_name_or_path": config.model.name_or_path
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(peft_config, f, indent=2)
        
    print(f"✅ LoRA adapter prepared at: {adapter_dir}")
    return adapter_dir

def _save_as_full_model(config, state_dict, output_dir):
    save_path = os.path.join(output_dir, "merged_model")
    
    # [기존 로직] 이미 존재하면 스킵
    has_bin = os.path.exists(os.path.join(save_path, "pytorch_model.bin"))
    has_safe = os.path.exists(os.path.join(save_path, "model.safetensors"))
    is_sharded = os.path.exists(os.path.join(save_path, "pytorch_model.bin.index.json")) or \
                 os.path.exists(os.path.join(save_path, "model.safetensors.index.json"))

    if (has_bin or has_safe or is_sharded) and os.path.exists(os.path.join(save_path, "config.json")):
        print(f"✅ Merged model already exists at: {save_path}")
        # ★ 디버깅을 위해: 이미 있어도 문제가 있다면 지우고 다시 만들어야 하므로
        # 확실치 않으면 이 폴더를 수동으로 삭제하고 돌리세요.
        return save_path

    print("🛠 Converting policy.pt to Full Merged Model...")
    base_model_id = config.model.name_or_path
    
    print(f"   > Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print("   > Applying weights...")
    cleaned_st = _clean_state_dict_keys(state_dict)
    
    # =========================================================
    # [수정] load_state_dict의 결과를 받아서 확인하는 로직 추가
    # =========================================================
    load_result = model.load_state_dict(cleaned_st, strict=False)
    
    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys
    
    # 로그 출력
    print(f"   > Load Results: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
    
    # [중요] 핵심 가중치가 빠졌는지 확인
    # 보통 rotary_emb.inv_freq 같은 버퍼 몇 개는 빠져도 되지만, 
    # layers, q_proj, v_proj 등이 빠지면 심각한 문제입니다.
    if len(missing_keys) > 0:
        print(f"⚠️  Missing Keys Example (Top 5): {missing_keys[:5]}")
        
        # 만약 전체 레이어가 다 빠졌다면 강제 종료하거나 경고
        if any("layers" in k for k in missing_keys):
            print("\n" + "!"*50)
            print("😱 CRITICAL ERROR: 핵심 레이어 가중치가 로드되지 않았습니다!")
            print("   policy.pt의 키 이름과 모델의 키 이름이 매칭되지 않습니다.")
            print("   _clean_state_dict_keys 함수를 수정해야 합니다.")
            print("!"*50 + "\n")
            # 필요시 raise ValueError("Weight mismatch detected")

    if len(unexpected_keys) > 0:
        print(f"⚠️  Unexpected Keys Example (Top 5): {unexpected_keys[:5]}")

    print(f"   > Saving to disk: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return save_path

# ==========================================
# [Main Export Function] - 여기가 핵심 수정 부분
# ==========================================

def prepare_weights_for_vllm(checkpoint_dir):
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    parent_dir = os.path.dirname(checkpoint_dir)
    config_path = os.path.join(parent_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    enable_lora = False
    if "lora" in config and hasattr(config.lora, "enabled"):
        enable_lora = config.lora.enabled

    # 1. LoRA인 경우
    if enable_lora:
        adapter_path = os.path.join(checkpoint_dir, "adapter")
        
        # [수정] .bin 뿐만 아니라 .safetensors도 확인합니다!
        has_bin = os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))
        has_safetensors = os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors"))
        
        # 둘 다 없으면 변환 시도
        if not (has_bin or has_safetensors):
            print("⚠️ No adapter file found (bin/safetensors). Looking for policy.pt...")
            policy_path = os.path.join(checkpoint_dir, "policy.pt")
            if os.path.exists(policy_path):
                state_dict = _load_state_dict_from_file(policy_path)
                _save_as_lora(config, state_dict, checkpoint_dir)
            else:
                print("❌ policy.pt not found either. Using base model ONLY.")
                return config.model.name_or_path, False, None
        else:
            print(f"✅ Found existing adapter (Safetensors/Bin) at: {adapter_path}")
        
        return config.model.name_or_path, True, adapter_path

    # 2. Full Fine-tuning인 경우
    else:
        merged_path = os.path.join(checkpoint_dir, "merged_model")
        
        # [수정] 병합 모델 내부도 safetensors 등 다양하게 확인
        # 간단히 config.json과 모델 파일 하나라도 있으면 존재하는 것으로 간주
        exists = os.path.exists(merged_path) and os.path.exists(os.path.join(merged_path, "config.json"))
        
        if not exists:
            policy_path = os.path.join(checkpoint_dir, "policy.pt")
            if os.path.exists(policy_path):
                state_dict = _load_state_dict_from_file(policy_path)
                _save_as_full_model(config, state_dict, checkpoint_dir)
            else:
                print("⚠️ policy.pt not found. Using raw base model.")
                return config.model.name_or_path, False, None
                
        return merged_path, False, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()
    
    m, l, p = prepare_weights_for_vllm(args.checkpoint_dir)
    print(f"\nResult:\nBase Model: {m}\nUse LoRA: {l}\nAdapter Path: {p}")
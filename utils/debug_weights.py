import torch
import os
import argparse

def inspect_weights(checkpoint_dir):
    policy_path = os.path.join(checkpoint_dir, "policy.pt")
    if not os.path.exists(policy_path):
        print("❌ policy.pt not found")
        return

    print(f"🔍 Inspecting: {policy_path}")
    state_dict = torch.load(policy_path, map_location="cpu")
    
    # 래핑 벗기기
    if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
    elif "model" in state_dict: state_dict = state_dict["model"]
    elif "state" in state_dict: state_dict = state_dict["state"]

    # 1. 키 이름 확인 (SCB 등이 있는지)
    keys = list(state_dict.keys())
    print(f"\n📊 Total keys: {len(keys)}")
    
    # FP8 관련 키 검색
    scb_keys = [k for k in keys if "SCB" in k or "weight_format" in k]
    print(f"⚠️ FP8/TE specific keys found: {len(scb_keys)}")
    if scb_keys:
        print(f"   Example: {scb_keys[:3]}")

    # 2. 실제 가중치 값 분석 (q_proj 기준)
    target_key = None
    for k in keys:
        if "layers.0.self_attn.q_proj.weight" in k and "SCB" not in k and "weight_format" not in k:
            target_key = k
            break
    
    if target_key:
        weight = state_dict[target_key]
        print(f"\n🔬 Analyzing weight: {target_key}")
        print(f"   Type: {weight.dtype}")
        print(f"   Shape: {weight.shape}")
        
        if torch.is_tensor(weight):
            # 텐서 통계
            w_float = weight.float()
            print(f"   Min: {w_float.min().item()}")
            print(f"   Max: {w_float.max().item()}")
            print(f"   Mean: {w_float.mean().item()}")
            print(f"   First 10 values: {w_float.flatten()[:10].tolist()}")
            
            # 3. SCB가 있다면 SCB 값도 확인
            scb_key = target_key.replace(".weight", ".SCB")
            if "_orig_mod." in scb_key: 
                 # 키 매칭을 위해 시도
                 pass 
            
            # 정확한 매칭 찾기
            found_scb_key = None
            for k in keys:
                if k.endswith("layers.0.self_attn.q_proj.SCB"):
                    found_scb_key = k
                    break
            
            if found_scb_key:
                scb = state_dict[found_scb_key]
                print(f"\n⚖️ Found paired SCB: {found_scb_key}")
                print(f"   Shape: {scb.shape}")
                print(f"   Values (First 5): {scb.flatten()[:5].tolist()}")

    else:
        print("❌ Could not find 'layers.0.self_attn.q_proj.weight'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()
    inspect_weights(args.checkpoint_dir)
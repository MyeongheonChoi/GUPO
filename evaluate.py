import torch
import hydra
import os
import tqdm
import wandb
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from transformers import BitsAndBytesConfig

# --- 1. 학습 스크립트에서 필요한 모듈들 임포트 ---
# (경로는 실제 프로젝트 구조에 맞게 수정해야 할 수 있습니다)
from trainers.gupo_trainers import GUPOTrainer
from preference_datasets import get_batch_iterator
from utils import (                        
    rank0_print,
    get_local_dir,
    slice_and_move_batch_for_device,
    formatted_dict,
    disable_dropout
)
import transformers
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# (train.py에서 모델 로드에 사용하던 다른 함수들도 필요시 임포트)


@hydra.main(config_path="outputs/gupo_joint", config_name="config") # train.py와 동일한 config 사용
def main(config: DictConfig):
    """
    저장된 체크포인트를 로드하여 MLP Beta 및 기타 지표를 평가합니다.
    """
    
    # 1. 모델 및 토크나이저 로드
    print('building policy')
    model_kwargs = {'device_map': 'balanced'}
    if config.model.policy_quantization == '8bit':
        print('using 8-bit quantization for policy model')
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model_kwargs['quantization_config'] = bnb_config
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, 
        cache_dir=get_local_dir(config.local_dirs), 
        low_cpu_mem_usage=True, 
        dtype=policy_dtype, 
        **model_kwargs
    )
    
    if config.lora.enabled:
        print('applying LoRA adapters')
        if getattr(policy, 'is_loaded_in_8bit', False) or getattr(policy, 'is_loaded_in_4bit', False):
            policy = prepare_model_for_kbit_training(policy)
            
        lora_config = LoraConfig(
            r=config.lora.r,
            target_modules=list(config.lora.target_modules),
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        policy = get_peft_model(policy, lora_config)
        policy.print_trainable_parameters()
    
    disable_dropout(policy)

    print('building reference model')
    reference_model_dtype = getattr(torch, config.model.reference_dtype)
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, 
        cache_dir=get_local_dir(config.local_dirs), 
        low_cpu_mem_usage=True, 
        dtype=reference_model_dtype, 
        **model_kwargs
    )
    disable_dropout(reference_model)

    # 2. BasicTrainers 인스턴스 생성
    #    (BasicTrainers의 __init__이 mlp, eval_batches 등을 생성)
    checkpoint_dir = "/home/mhchoi/GUPO/outputs/gupo_joint"
    if not checkpoint_dir:
        raise ValueError("config에 'checkpoint_dir'을(를) 지정해야 합니다.")

    rank0_print("Initializing BasicTrainers...")
    trainer = GUPOTrainer(
        policy=policy,
        config=config,
        seed=config.seed,
        run_dir=f"gupo_mlp_evaluation_{checkpoint_dir.split('/')[-1]}", # 임시 실행 디렉토리
        reference_model=reference_model,
    )

    # 3. 체크포인트 로드
    

    rank0_print(f"Loading checkpoints from {checkpoint_dir}...")
    
    # Policy (LoRA 어댑터) 체크포인트 로드
    policy_checkpoint_path = os.path.join(checkpoint_dir, 'policy.pt')
    if os.path.exists(policy_checkpoint_path):
        policy_checkpoint = torch.load(policy_checkpoint_path, map_location='cpu')
        trainer.policy.load_state_dict(policy_checkpoint['state'])
        rank0_print(f"Loaded policy checkpoint from step {policy_checkpoint.get('step_idx', 'N/A')}")
    else:
        raise FileNotFoundError(f"Warning: policy.pt not found in {checkpoint_dir}.")

    # MLP 체크포인트 로드
    mlp_checkpoint_path = os.path.join(checkpoint_dir, 'mlp.pt')
    if os.path.exists(mlp_checkpoint_path):
        mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location='cpu')
        trainer.mlp.load_state_dict(mlp_checkpoint['state'])
        rank0_print(f"Loaded MLP checkpoint from step {mlp_checkpoint.get('step_idx', 'N/A')}")
    else:
        raise FileNotFoundError(f"mlp.pt not found in {mlp_checkpoint_path}. MLP 평가에 필수입니다.")

    # 4. 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # trainer.policy.to(device)
    # trainer.reference_model.to(device)
    trainer.mlp.to(device)
    
    # 5. MLP 베타 평가 실행 (vLLM 없이)
    #    (train 함수에서 복사해 온 evaluate_mlp 함수 사용)
    mean_eval_metrics = trainer.evaluate_mlp()
    
    rank0_print("--- 📊 Final Evaluation Metrics ---")
    rank0_print(formatted_dict(mean_eval_metrics))
    rank0_print("--- ---------------------------- ---")
    

    rank0_print("✅ MLP Beta evaluation complete.")

if __name__ == "__main__":
    main()
import torch
import hydra
import os
import tqdm
import wandb
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

# --- 1. 학습 스크립트에서 필요한 모듈들 임포트 ---
# (경로는 실제 프로젝트 구조에 맞게 수정해야 할 수 있습니다)
from trainers.gupo_trainers import GUPOTrainer
from preference_datasets import get_batch_iterator
from utils import (                        
    rank0_print,
    get_local_dir,
    slice_and_move_batch_for_device,
    formatted_dict
)
import transformers
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# (train.py에서 모델 로드에 사용하던 다른 함수들도 필요시 임포트)


def load_model_and_tokenizer(config: DictConfig):
    """
    학습 스크립트와 동일한 로직으로 모델과 토크나이저를 로드합니다.
    """
    rank0_print(f"Loading tokenizer from {config.model.tokenizer_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path or config.model.name_or_path
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rank0_print(f"Loading policy model: {config.model.name_or_path}")
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=torch.bfloat16,  # bfloat16 사용
        # (bitsandbytes 4/8bit 로드 설정이 있다면 여기에 추가)
    )

    if config.lora.enabled:
        rank0_print("Applying LoRA adapters to policy model...")
        
        # PEFT JSON 직렬화 오류 방지 (ListConfig -> list)
        target_modules_list = list(config.lora.target_modules)
        
        lora_config = LoraConfig(
            r=config.lora.r,
            target_modules=target_modules_list,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        policy = get_peft_model(policy, lora_config)
        rank0_print("LoRA applied. Trainable parameters:")
        policy.print_trainable_parameters()

    rank0_print(f"Loading reference model: {config.model.name_or_path}")
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=torch.bfloat16, # bfloat16 사용
    )
    
    return policy, reference_model, tokenizer


@hydra.main(config_path="configs", config_name="config") # train.py와 동일한 config 사용
def main(config: DictConfig):
    """
    저장된 체크포인트를 로드하여 MLP Beta 및 기타 지표를 평가합니다.
    """
    
    # 1. 모델 및 토크나이저 로드
    policy, reference_model, tokenizer = load_model_and_tokenizer(config)
    
    # 2. BasicTrainers 인스턴스 생성
    #    (BasicTrainers의 __init__이 mlp, eval_batches 등을 생성)
    rank0_print("Initializing BasicTrainers...")
    trainer = GUPOTrainer(
        policy=policy,
        config=config,
        seed=config.seed,
        run_dir=f"evaluation_run_{config.checkpoint_dir.split('/')[-1]}", # 임시 실행 디렉토리
        reference_model=reference_model,
        tokenizer=tokenizer, # __init__에서 필요하면 전달
        rank=0,         # 단일 GPU 평가 가정
        world_size=1
    )

    # 3. 체크포인트 로드
    checkpoint_dir = config.checkpoint_dir
    if not checkpoint_dir:
        raise ValueError("config에 'checkpoint_dir'을(를) 지정해야 합니다.")

    rank0_print(f"Loading checkpoints from {checkpoint_dir}...")
    
    # Policy (LoRA 어댑터) 체크포인트 로드
    policy_checkpoint_path = os.path.join(checkpoint_dir, 'policy.pt')
    if os.path.exists(policy_checkpoint_path):
        policy_checkpoint = torch.load(policy_checkpoint_path, map_location='cpu')
        trainer.policy.load_state_dict(policy_checkpoint['state'])
        rank0_print(f"Loaded policy checkpoint from step {policy_checkpoint.get('step_idx', 'N/A')}")
    else:
        rank0_print(f"Warning: policy.pt not found in {checkpoint_dir}. Using base LoRA.")

    # MLP 체크포인트 로드
    mlp_checkpoint_path = os.path.join(checkpoint_dir, 'mlp.pt')
    if os.path.exists(mlp_checkpoint_path):
        mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location='cpu')
        trainer.mlp.load_state_dict(mlp_checkpoint['state'])
        rank0_print(f"Loaded MLP checkpoint from step {mlp_checkpoint.get('step_idx', 'N/A')}")
    else:
        raise FileNotFoundError(f"mlp.pt not found in {checkpoint_dir}. MLP 평가에 필수입니다.")

    # 4. 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.policy.to(device)
    trainer.reference_model.to(device)
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
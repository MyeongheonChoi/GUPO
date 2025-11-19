import torch
import hydra
import os
import argparse
from omegaconf import OmegaConf
from transformers import BitsAndBytesConfig

# --- 1. 학습 스크립트에서 필요한 모듈들 임포트 ---
# (경로는 실제 프로젝트 구조에 맞게 수정해야 할 수 있습니다)
from trainers.gupo_trainers import GUPOTrainer
from utils import (                        
    rank0_print,
    get_local_dir,
    disable_dropout
)
import transformers

# (train.py에서 모델 로드에 사용하던 다른 함수들도 필요시 임포트)

def save_extreme_cases(df_results, checkpoint_dir, n):
    df_results['avg'] = (df_results['chosen_beta_mlp']+ df_results['rejected_beta_mlp']) / 2
    df_min = df_results.sort_values(by='avg').head(n)
    df_max = df_results.sort_values(by='avg', ascending=False).head(n)

    with open(os.path.join(checkpoint_dir, "beta_evaluation_extreme_cases.txt"), "w") as f:
        f.write("===== Top {} Minimum Average Beta Cases =====\n\n".format(n))
        for i in df_min[['prompt', 'chosen_response', 'rejected_response', 'avg']].itertuples():
            f.write("Average Beta: {:.4f}\n".format(i.avg))
            f.write(f"Prompt: {i.prompt}\nChosen : {i.chosen_response}\nRejected : {i.rejected_response}\n")
            f.write('=' * 50 + '\n')

        f.write("\n===== Top {} Maximum Average Beta Cases =====\n\n".format(n))
        for i in df_max[['prompt', 'chosen_response', 'rejected_response', 'avg']].itertuples():
            f.write("Average Beta: {:.4f}\n".format(i.avg))
            f.write(f"Prompt: {i.prompt}\nChosen : {i.chosen_response}\nRejected : {i.rejected_response}\n")
            f.write('=' * 50 + '\n')
    
    print("Extreme cases saved to beta_evaluation_extreme_cases.txt")

def main(args):

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    print(f"Using checkpoint directory: {checkpoint_dir}")

    n = args.n

    parent_dir = os.path.dirname(checkpoint_dir)

    config_dir = os.path.join(parent_dir, "config.yaml")
    config = OmegaConf.load(config_dir)

    print('building policy base model')
    model_kwargs = {}

    if config.model.policy_quantization == '8bit':
        print('using 8-bit quantization for policy model')
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs['quantization_config'] = bnb_config
        
    policy_dtype = getattr(torch, config.model.policy_dtype)
    
    # 1-1. Base Model 로드
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, 
        cache_dir=get_local_dir(config.local_dirs), 
        low_cpu_mem_usage=True, 
        dtype=policy_dtype, 
        **model_kwargs
    )
    
    # 1-2. LoRA vs Full Fine-tuning 분기 처리
    if config.lora.enabled:
        print('Loading LoRA adapter...')
        adapter_path = os.path.join(checkpoint_dir, 'adapter')
        print(adapter_path)
        
        if os.path.exists(adapter_path):
            from peft import PeftModel
            policy = PeftModel.from_pretrained(policy, adapter_path)
            print(f"✅ Loaded LoRA adapter from {adapter_path}")
        else:
            raise FileNotFoundError(f"LoRA enabled but adapter not found at {adapter_path}")
            
    else:
        print('Loading full policy weights...')
        policy_checkpoint_path = os.path.join(checkpoint_dir, 'policy.pt')
        
        if os.path.exists(policy_checkpoint_path):
            state_dict = torch.load(policy_checkpoint_path, map_location='cpu', weights_only=True)['state']
            policy.load_state_dict(state_dict)
            print(f"✅ Loaded full policy weights from {policy_checkpoint_path}")
        else:
            raise FileNotFoundError(f"policy.pt not found at {policy_checkpoint_path}")

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

    rank0_print("Initializing BasicTrainers...")
    trainer = GUPOTrainer(
        policy=policy,
        config=config,
        seed=config.seed,
        run_dir=f"gupo_mlp_evaluation_{checkpoint_dir.split('/')[-1]}", 
        reference_model=reference_model,
    )

    mlp_checkpoint_path = os.path.join(checkpoint_dir, 'mlp.pt')
    if os.path.exists(mlp_checkpoint_path):
        mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location='cpu', weights_only=True)
        trainer.mlp.load_state_dict(mlp_checkpoint['state'])
        rank0_print(f"✅ Loaded MLP checkpoint from step {mlp_checkpoint.get('step_idx', 'N/A')}")
    else:
        raise FileNotFoundError(f"mlp.pt not found in {checkpoint_dir}. MLP 평가에 필수입니다.")
    
    trainer.mlp.to(trainer.policy.device)

    rank0_print("Generating Beta Data Table...")
    
    df_results = trainer.generate_beta_dataset()
    save_path = os.path.join(checkpoint_dir, "mlp_beta_evaluation_results.csv")
    df_results.to_csv(save_path, index=False, encoding='utf-8-sig') # utf-8-sig는 엑셀 한글 깨짐 방지
    
    rank0_print(f"✅ Data table saved to: {save_path}")
    
    print(df_results.head())

    print("\n--- Beta Statistics ---")
    print(df_results[['chosen_beta_mlp', 'rejected_beta_mlp']].describe())

    save_extreme_cases(df_results, checkpoint_dir, n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLP Beta from Checkpoint")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True, 
        help="Path to the checkpoint directory containing policy.pt and mlp.pt"
    )
    parser.add_argument(
        "--n", 
        type=int, 
        default=10, 
        help="Number of extreme cases to log for min and max average beta"
    )
    args = parser.parse_args()
    
    main(args)
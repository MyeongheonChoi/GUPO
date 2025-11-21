import argparse
import os
import json
import torch
from omegaconf import OmegaConf
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main(args):
    # 1. 학습 Config 로드
    # 사용자가 입력한 체크포인트 경로를 절대 경로로 변환
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    parent_dir = os.path.dirname(checkpoint_dir)
    config_dir = os.path.join(parent_dir, "config.yaml")
    config = OmegaConf.load(config_dir)
    

    base_model_path = config.model.name_or_path
    print(f"🚀 Initializing vLLM Engine with base model: {base_model_path}")
    
    # LoRA 설정 확인
    enable_lora = config.lora.enabled
    
    llm = LLM(
        model=base_model_path,
        enable_lora=enable_lora,
        dtype="bfloat16",
        seed=config.seed,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_len if args.max_len else config.model.get('max_length', 2048), # Config 없으면 args 사용
    )

    # 샘플링 파라미터 (생성 시에만 쓰이는 설정이므로 args로 받음)
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        skip_special_tokens=True,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[llm.get_tokenizer().eos_token_id]
    )

    # LoRA 요청 객체 생성
    lora_request = None
    if enable_lora:
        # 어댑터 경로는 체크포인트 폴더 안의 'adapter' 폴더로 자동 지정
        adapter_path = os.path.join(checkpoint_dir, 'adapter')
        
        if not os.path.exists(adapter_path):
             raise FileNotFoundError(f"❌ LoRA is enabled in config, but adapter not found at: {adapter_path}")

        print(f"✅ LoRA Adapter will be applied from: {adapter_path}")
        lora_request = LoRARequest("gupo_adapter", 1, adapter_path)

    # ------------------------------------------------------------------
    # 3. 테스트 데이터셋 로드 (생성 대상)
    # ------------------------------------------------------------------
    dataset_name = args.dataset_name or config.datasets[0] # args가 없으면 학습 데이터셋 사용 (선택)
    print(f"📂 Loading dataset: {dataset_name} (split: {args.split})")
    
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_name, split=args.split)
    else:
        dataset = load_dataset(dataset_name, split=args.split)

    # 프롬프트 컬럼 찾기
    prompt_col = args.prompt_column
    if prompt_col not in dataset.column_names:
        if 'prompt' in dataset.column_names: prompt_col = 'prompt'
        elif 'instruction' in dataset.column_names: prompt_col = 'instruction'
        else: raise ValueError(f"Dataset columns {dataset.column_names} do not contain '{prompt_col}' key.")
            
    prompts = dataset[prompt_col]
    print(f"📊 Total samples to generate: {len(prompts)}")

    # ------------------------------------------------------------------
    # 4. 문장 생성 & 저장
    # ------------------------------------------------------------------
    print("⚡ Starting generation...")
    outputs = llm.generate(
        prompts, 
        sampling_params, 
        lora_request=lora_request
    )

    results = []
    for output in outputs:
        results.append({
            "prompt": output.prompt,
            "generated_response": output.outputs[0].text
        })

    # 저장 경로: 체크포인트 폴더 안에 'generation_result.jsonl'로 저장
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(checkpoint_dir, "generation_result.jsonl")

    print(f"💾 Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    base_name = os.path.splitext(output_path)[0]
    config_save_path = f"{base_name}_config.json"
    
    print(f"⚙️ Saving generation config to {config_save_path}...")
    
    # args 객체를 딕셔너리로 변환하여 저장
    generation_config = vars(args)
    
    # (선택 사항) 보기 좋게 저장된 절대 경로들도 추가해주면 좋습니다
    generation_config['saved_checkpoint_dir_abs'] = checkpoint_dir
    
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(generation_config, f, indent=4, ensure_ascii=False)
    # ▲▲▲ 설정 저장 완료 ▲▲▲

    print("✅ Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using vLLM with trained config")
    
    # 필수: 체크포인트 경로 (여기에 config.yaml과 adapter 폴더가 있어야 함)
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the checkpoint directory (e.g., outputs/exp/step-1000)")
    
    # 선택: 데이터셋 (지정 안 하면 config의 학습 데이터셋을 쓸 수도 있음)
    parser.add_argument("--dataset_name", type=str, default="anthropic/hh-rlhf", help="Dataset to generate responses for")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts")
    
    # 선택: 생성 파라미터
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--max_len", type=int, default=None, help="Max context length (default: use config or 2048)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p")
    
    # 선택: 출력 파일명 (기본값: checkpoint_dir/generation_result.jsonl)
    parser.add_argument("--output_file", type=str, default=None, help="Custom output file path")

    args = parser.parse_args()
    main(args)
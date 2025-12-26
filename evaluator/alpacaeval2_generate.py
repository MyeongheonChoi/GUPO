import json
import argparse
import os
import sys
import json
import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset
from vllm import LLM, SamplingParams
# from alpaca_eval import datasets

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.convert_to_hf import prepare_weights_for_vllm


# 1. 설정

def main(args):
    # 2. 데이터셋 로드
    eval_set = load_dataset('json', data_files='./evaluator/alpaca_eval_gpt4_baseline.json', split='train')
    # print(eval_set)
    instructions = [example['instruction'] for example in eval_set]
    
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    parent_dir = os.path.dirname(checkpoint_dir)
    config_dir = os.path.join(parent_dir, "config.yaml")
    config = OmegaConf.load(config_dir)

    print("🔍 Checking and preparing model weights...")
    model_path, use_lora, lora_path = prepare_weights_for_vllm(checkpoint_dir)
    print(f"🚀 Initializing vLLM Engine")
    print(f"   - Base Model: {model_path}")

    llm = LLM(
        model=model_path,
        enable_lora=use_lora,
        dtype="bfloat16",
        seed=config.seed,
        gpu_memory_utilization=0.3,
        max_model_len=args.max_len if args.max_len else config.model.get('max_length', 2048),
    )
    stop_words = ["\nHuman:", "\n\nHuman:", "Human:", "\nUser:", "\n\nUser:"]
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        skip_special_tokens=True,
        max_tokens=args.max_new_tokens,
        stop=stop_words,
        stop_token_ids=[llm.get_tokenizer().eos_token_id]
    )



    # 4. 입력을 채팅 템플릿에 맞게 변환 (vLLM은 자동 처리가 아니므로 수동 처리 필요할 수 있음)
    # 간단한 처리를 위해 여기서는 템플릿 없이 진행하거나, 토크나이저를 별도로 불러와 처리해야 합니다.
    # 아래는 일반적인 Llama-3 포맷 예시입니다.
    formatted_prompts = [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" 
        for inst in instructions
    ]
    # formatted_prompts = [
    #     f"\n\nHuman: {inst}\n\nAssistant: " 
    #     for inst in instructions
    # ]

    # 5. 고속 생성
    outputs = llm.generate(formatted_prompts, sampling_params)

    # 6. 결과 정리
    results = []
    for instruction, output in zip(instructions, outputs):
        generated_text = output.outputs[0].text
        results.append({
            "instruction": instruction,
            "output": generated_text,
            "generator": checkpoint_dir
        })

    # 7. 저장
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, args.output_name + '.json'), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and Evaluate responses")
    
    parser.add_argument("--checkpoint_dir", type=str, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="./evaluator/alpacaeval2", help="Directory to save the results. If not set, saves in checkpoint_dir")
    parser.add_argument("--output_name", type=str, default="gupo", help="Directory to save the results. If not set, saves in checkpoint_dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and generation")
    
    parser.add_argument("--num_eval_samples", type=int, default=-1, help="Number of prompts to randomly sample from dataset")
    
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()
    main(args)

    # env $(grep -v '^#' .env | xargs) uvx --python 3.11 --with "datasets==2.19.1" --with setuptools alpaca_eval --model_outputs {result of alpacaeval2_generate.py} --annotators_config 'alpaca_eval_gpt4_turbo_fn'

import argparse
import os
import json
import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.preference_datasets import extract_anthropic_prompt
from utils.convert_to_hf import prepare_weights_for_vllm


def main(args):
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    parent_dir = os.path.dirname(checkpoint_dir)
    config_dir = os.path.join(parent_dir, "config.yaml")
    config = OmegaConf.load(config_dir)
    
    if OmegaConf.select(config, "loss.residual") is not None:
        pass 

    else:
        if "residual" in config.exp_name:
            config.loss.residual = True
        else:
            config.loss.residual = False

    print("🔍 Checking and preparing model weights...")
    model_path, use_lora, lora_path = prepare_weights_for_vllm(checkpoint_dir)
    print(f"🚀 Initializing vLLM Engine")
    print(f"   - Base Model: {model_path}")

    llm = LLM(
        model=model_path,
        enable_lora=use_lora,
        dtype="bfloat16",
        seed=config.seed,
        gpu_memory_utilization=0.9,
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

    # LoRA 요청 객체 생성
    lora_request = None
    if use_lora and lora_path:
        print(f"✅ LoRA Adapter will be applied from: {lora_path}")
        lora_request = LoRARequest("gupo_adapter", 1, lora_path)

    dataset_name = args.dataset_name or config.datasets[0] 
    print(f"📂 Loading dataset: {dataset_name} (split: {args.split})")
    
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_name, split=args.split, data_dir='harmless-base')
    else:
        dataset = load_dataset(dataset_name, split=args.split, data_dir='harmless-base')
        
    # if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
    #     dataset = load_dataset("json", data_files=dataset_name, split=args.split)
    # else:
    #     dataset = load_dataset(dataset_name, split=args.split)

    # 프롬프트 컬럼 찾기
    prompts = []
    for row in tqdm.tqdm(dataset, desc='Processing HH'):
        prompt = extract_anthropic_prompt(row['chosen'])
        prompts.append(prompt)
    # prompt_col = args.prompt_column
    # if prompt_col not in dataset.column_names:
    #     if 'prompt' in dataset.column_names: prompt_col = 'prompt'
    #     elif 'instruction' in dataset.column_names: prompt_col = 'instruction'
    #     else: raise ValueError(f"Dataset columns {dataset.column_names} do not contain '{prompt_col}' key.")
            
    # prompts = dataset[prompt_col]
    
    print(f"📊 Total samples to generate: {len(prompts)}")

    print("⚡ Starting generation...")
    outputs = llm.generate(
        prompts, 
        sampling_params, 
        lora_request=lora_request
    )

    results = []
    for output in outputs:
        # results.append({
        #     "prompt": output.prompt,
        #     "generated_response": output.outputs[0].text
        # })
        results.append({
            "prompt": output.prompt,
            "generated_response": [output.outputs[i].text for i in range(len(output.outputs))],
        })    
    
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(checkpoint_dir, "generation_result.jsonl")
        # output_path = os.path.join(checkpoint_dir, "generation_result_full.jsonl")
        # output_path = os.path.join(checkpoint_dir, "generation_result_3.jsonl")

    print(f"💾 Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    base_name = os.path.splitext(output_path)[0]
    config_save_path = f"{base_name}_config.json"
    
    print(f"⚙️ Saving generation config to {config_save_path}...")
    
    generation_config = vars(args)
    
    generation_config['saved_checkpoint_dir_abs'] = checkpoint_dir
    
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(generation_config, f, indent=4, ensure_ascii=False)

    print("✅ Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using vLLM with trained config")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the checkpoint directory (e.g., outputs/exp/step-1000)")
    
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf", help="Dataset to generate responses for")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts")
    
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--max_len", type=int, default=None, help="Max context length (default: use config or 2048)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p")
    
    parser.add_argument("--output_file", type=str, default=None, help="Custom output file path")

    args = parser.parse_args()
    main(args)
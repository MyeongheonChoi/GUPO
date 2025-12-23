import argparse
import os
import json
import tqdm
import random
import torch
import numpy as np
import gc
from omegaconf import OmegaConf
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.preference_datasets import extract_anthropic_prompt
from utils.convert_to_hf import prepare_weights_for_vllm

HELPFUL_RM_PATH = "Ray2333/gpt2-large-helpful-reward_model"
HARMLESS_RM_PATH = "Ray2333/gpt2-large-harmless-reward_model"
ARMO_RM_PATH = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_conversation(text):
    """
    'Human: ... \n\nAssistant: ...' 형태의 텍스트를
    [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}] 리스트로 변환
    """
    messages = []
    
    # "Human: " 앞에 개행이 없을 경우를 대비해 처리
    if text.startswith("Human:"):
        text = "\n\n" + text
        
    # \n\nHuman: 또는 \n\nAssistant: 로 턴이 구분됨
    parts = text.split("\n\n")
    
    for part in parts:
        part = part.strip()
        if part.startswith("Human:"):
            content = part.replace("Human:", "").strip()
            messages.append({"role": "user", "content": content})
        elif part.startswith("Assistant:"):
            content = part.replace("Assistant:", "").strip()
            messages.append({"role": "assistant", "content": content})
            
    return messages

class ArmoRMPipeline:
    def __init__(self, model_id, device, torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=True, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages):
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.rewards.cpu().float()
            helpfulness = score[0][9]
            helpfulness1 = score[0][0]
            harmless = score[0][10]

        return helpfulness, helpfulness1, harmless

def get_reward_scores(samples, rm, device="cuda"):
    print("⚖️ Loading Reward Models for evaluation...")

    if rm == 'gpt2':
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Helpful RM
        h_tokenizer = AutoTokenizer.from_pretrained(HELPFUL_RM_PATH)
        h_model = AutoModelForSequenceClassification.from_pretrained(HELPFUL_RM_PATH, torch_dtype=torch.bfloat16).to(device)
        h_model.eval()
        
        # Harmless RM
        hl_tokenizer = AutoTokenizer.from_pretrained(HARMLESS_RM_PATH)
        hl_model = AutoModelForSequenceClassification.from_pretrained(HARMLESS_RM_PATH, torch_dtype=torch.bfloat16).to(device)
        hl_model.eval()

        print("📊 Scoring responses...")
        scored_results = []
        
        with torch.no_grad():
            for item in tqdm.tqdm(samples, desc="Scoring"):
                prompt = item['prompt']
                response = item['generated_response']
                full_text = prompt + response

                inputs_h = h_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
                inputs_hl = hl_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(device)

                helpful_score = h_model(**inputs_h).logits[0].item()
                harmless_score = hl_model(**inputs_hl).logits[0].item()

                avg_score = (helpful_score + harmless_score) / 2
                
                item['helpful_score'] = helpful_score
                item['harmless_score'] = harmless_score
                item['reward_score'] = avg_score
                scored_results.append(item)
                
        del h_model, hl_model, h_tokenizer, hl_tokenizer
        torch.cuda.empty_cache()
    
    elif rm == 'armorm':
        import transformers.models.llama.modeling_llama

        if not hasattr(transformers.models.llama.modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
            transformers.models.llama.modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        scored_results = []
        armorm_pipeline = ArmoRMPipeline(ARMO_RM_PATH, device=device)
        for item in tqdm.tqdm(samples, desc="Scoring"):
            prompt = item['prompt']
            generated_response = item['generated_response']
            generated_messages = parse_conversation(prompt + generated_response)

            help1, help2, harmless = armorm_pipeline(generated_messages)
            
            helpful = (help1 + help2) / 2
            avg_score = (helpful + harmless) / 2
            item['helpful_score'] = helpful.item()
            item['harmless_score'] = harmless.item()
            item['reward_score'] = avg_score.item()
            scored_results.append(item)

    return scored_results

def main(args):
    set_seed(args.seed)
    print(args.generate)
    if args.generate:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        parent_dir = os.path.dirname(checkpoint_dir)
        config_dir = os.path.join(parent_dir, "config.yaml")
        config = OmegaConf.load(config_dir)

        dataset_name = args.dataset_name or config.datasets[0]
        print(f"📂 Loading dataset: {dataset_name} (split: {args.split})")
        dataset = load_dataset(dataset_name, split=args.split)

        print("🔍 Processing prompts...")
        all_prompts = []
        for row in tqdm.tqdm(dataset, desc='Extracting Prompts'):
            prompt = extract_anthropic_prompt(row['chosen'])
            all_prompts.append(prompt)
        
        if args.num_eval_samples > 0 and args.num_eval_samples < len(all_prompts):
            print(f"🎲 Randomly sampling {args.num_eval_samples} prompts (Seed: {args.seed})")
            eval_prompts = random.sample(all_prompts, args.num_eval_samples)
        else:
            print(f"Using all {len(all_prompts)} prompts.")
            eval_prompts = all_prompts

        print("🔍 Checking and preparing model weights...")
        model_path, use_lora, lora_path = prepare_weights_for_vllm(checkpoint_dir)
        
        print(f"🚀 Initializing vLLM Engine")
        llm = LLM(
            model=model_path,
            enable_lora=use_lora,
            dtype="bfloat16",
            seed=args.seed,
            gpu_memory_utilization=0.5,
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

        lora_request = None
        if use_lora and lora_path:
            print(f"✅ LoRA Adapter will be applied from: {lora_path}")
            lora_request = LoRARequest("gupo_adapter", 1, lora_path)

        print(f"⚡ Starting generation for {len(eval_prompts)} prompts...")
        outputs = llm.generate(eval_prompts, sampling_params, lora_request=lora_request)

        generated_results = []
        for output in outputs:
            for idx, completion_output in enumerate(output.outputs):
                generated_results.append({
                    "prompt": output.prompt,
                    "generated_response": completion_output.text,
                    "output_index": idx 
                })

        print("🧹 Cleaning up vLLM to free GPU memory...")
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    else:
        output_path = os.path.join(
            args.output_dir,
            f"generation_result_seed{args.seed}_armorm.json"
        )
        print(f"♻️ Loading previously generated results from {output_path}...")
        generated_results = load_dataset(
            "json", data_files=output_path, split="train"
        ).to_list()

    final_results = get_reward_scores(generated_results, args.rm)

    if args.output_file:
        output_path = args.output_file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    else:
        if args.output_dir:
            save_dir = args.output_dir
        else:
            save_dir = checkpoint_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"generation_result_seed{args.seed}_{args.rm}.json"
        output_path = os.path.join(save_dir, filename)
    
    base, ext = os.path.splitext(output_path)
    summary_path = f"{base}_summary{ext}"

    print(f"💾 Saving detailed results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    avg_helpful = np.mean([x['helpful_score'] for x in final_results])
    avg_harmless = np.mean([x['harmless_score'] for x in final_results])
    avg_reward = np.mean([x['reward_score'] for x in final_results])

    by_index_stats = {}
    for i in range(args.n_samples):
        subset = [x for x in final_results if x.get('output_index') == i]
        
        if subset:
            by_index_stats[f"response_{i}"] = {
                "avg_reward": float(np.mean([x['reward_score'] for x in subset])),
                "avg_helpful": float(np.mean([x['helpful_score'] for x in subset])),
                "avg_harmless": float(np.mean([x['harmless_score'] for x in subset]))
            }
    
    summary_data = {
        "model_checkpoint": checkpoint_dir if args.generate else "N/A (loaded results)",
        "seed": args.seed,
        "num_samples_per_prompt": args.n_samples,
        "total_generated_samples": len(final_results),

        "overall_averages": {
            "reward": float(avg_reward),
            "helpful": float(avg_helpful),
            "harmless": float(avg_harmless)
        },
        
        "averages_by_output_index": by_index_stats,
        
        "generation_config": vars(args)
    }

    print(f"📝 Saving summary to {summary_path}...")
    print(f"   [Overall] Reward: {avg_reward:.4f}")

    for k, v in by_index_stats.items():
        print(f"   [{k}] Reward: {v['avg_reward']:.4f}")

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    print("✅ Evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and Evaluate responses")
    
    parser.add_argument("--checkpoint_dir", type=str, help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results/llama_joint_0.0", help="Directory to save the results. If not set, saves in checkpoint_dir")
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and generation")
    
    parser.add_argument("--num_eval_samples", type=int, default=-1, help="Number of prompts to randomly sample from dataset")
    
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_file", type=str, default=None)

    parser.add_argument("--rm", type=str, default=None, help = "Reward model to use for evaluation. Options: 'gpt2', 'armorm'")
    parser.add_argument("--generate", action='store_true', help="Whether to generate new responses or load existing ones")


    args = parser.parse_args()
    main(args)
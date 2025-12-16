from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os
import argparse


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
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
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
        verbosity = score[0][4]
        helpfulness = score[0][9]
        helpfulness1 = score[0][0]
        harmless = score[0][10]

        return verbosity, helpfulness, helpfulness1, harmless
    
def evaluate_armorm(output_path):
    import transformers.models.llama.modeling_llama

    if not hasattr(transformers.models.llama.modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
        transformers.models.llama.modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)

    # --- [4. 실행 및 점수 비교] ---
    # 데이터셋 로드 (테스트를 위해 5개만)
    dataset = load_dataset("json", data_files=output_path, split="train")

    results = []

    print("Processing...")

    for item in tqdm(dataset):
        # 1. 전체 대화(Prompt + Response)를 리스트로 변환
        prompt = item['prompt']
        generated_responses = item['generated_response']
        # item['rejected']는 Prompt + Rejected Response가 합쳐져 있음
        if len(generated_responses) == 1:
            generated_response = generated_responses[0]
            generated_messages = parse_conversation(prompt + generated_response)
            score = rm(generated_messages)[-1]
            results.append({
                "prompt": prompt, # 마지막 질문(Prompt)
                "generated_response": generated_response, # Chosen 답변 앞부분
                "reward": float(score),
            })

        else:
            for generated_response in generated_responses:
                generated_messages = parse_conversation(prompt + generated_response)
                score = rm(generated_messages)[-1]
                results.append({
                    "prompt": prompt, # 마지막 질문(Prompt)
                    "generated_response": generated_response, # Chosen 답변 앞부분
                    "reward": float(score),
                })



        # print(chosen_messages)
        # print('-'*50)
        # print(rejected_messages)
        # 2. 모델 점수 계산
        # rm 함수는 전체 리스트를 받아, '마지막 assistant 답변'에 대한 점수를 줍니다.

        # 3. 결과 저장 (검증을 위해 마지막 답변 텍스트도 같이 저장)
        

    # --- [5. 결과 출력] ---
    df = pd.DataFrame(results)

    # 보기 좋게 출력
    pd.set_option('display.max_colwidth', 50)
    print(df[['prompt', 'generated_response', 'reward']])

    # save_path = os.path.join(os.path.dirname(output_path), 'armorm_eval.csv')
    save_path = os.path.join(os.path.dirname(output_path), 'armorm_eval.csv')
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Save Successful {save_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MLP Beta from Checkpoint")
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to the generated response file (.../generated_result.jsonl)"
    )
    args = parser.parse_args()
    
    evaluate_armorm(args.output_path)
import json
import random
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="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")
model_name = "gpt-5-mini"

dataset = load_dataset("Anthropic/hh-rlhf", split="test")
dataset = dataset.select([i for i in range(10)])

def get_ambiguity(prompt, chosen_response, rejected_response, model_name):
    flip = random.choice([0, 1])
    if flip == 0:
        response_a, response_b = chosen_response, rejected_response
    else:
        response_a, response_b = rejected_response, chosen_response
    
    system_prompt = """
You are an expert AI evaluator.
Rate the **ambiguity** (difficulty of choice) between two responses based on helpfulness and harmlessness.
Higher score means higher ambiguity (harder to choose).

Scale (1-5):
1: Very clear difference (obvious winner)
2: Clear difference
3: Moderate difference
4: High ambiguity (hard to distinguish)
5: Indistinguishable / tie (maximum ambiguity)

Return JSON: {"reason": "Brief explanation...", "ambiguity_score": int, "better_response": "A" or "B" or "Tie"}
"""

    user_prompt = f"""
[Prompt]
{prompt}

[Response_A]
{response_a}

[Response_B]
{response_b}

Rate the ambiguity and specify your reason. Output JSON only.
"""

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format = {"type": "json_object"},
        )
    
        result = json.loads(completion.choices[0].message.content)
        ambiguity_score = result.get("ambiguity_score")
        predicted_winner = result.get("better_response")
        reason = result.get("reason")
        is_correct = False
        if flip == 0 and predicted_winner == "A": is_correct = True
        elif flip == 1 and predicted_winner == "B": is_correct = True

        return {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "ambiguity_score": ambiguity_score,
            "reason": reason,
            "flip": flip,
            "is_correct": is_correct,
        }
    
    except Exception as e:
        print(e)
        return None

results = []

for data in tqdm(dataset):
    chosen = data["chosen"]
    rejected = data["rejected"]

    split_token = "\n\nAssistant: "
    prompt = chosen.rsplit(split_token, 1)[0]+split_token
    chosen_response = chosen.rsplit(split_token, 1)[1]
    rejected_response = rejected.rsplit(split_token, 1)[1]
    
    result = get_ambiguity(prompt, chosen_response, rejected_response, model_name)
    results.append(result)

file_name = "ambiguity_results.json"
with open(file_name, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

csv_file_name = "ambiguity_results.csv"
df = pd.DataFrame(results)
df.to_csv(csv_file_name, index=False, encoding="utf-8-sig")

print(f"succesfully saved!")
import os
import random
import requests
from datasets import load_from_disk
from tqdm import tqdm
import re
import time

# === Config ===
DEEPSEEK_API_KEY = "sk-2377e5a6b92e4847b17ed9f51bdeebd9"
DATASET_DIR = os.path.expanduser("~/active_learning/processed_data/refind")
OUTPUT_DIR = "./selection_outputs/two_stage_chunk"
CHUNK_SIZE_2 = 5  # M: Agent 2 selects 1 out of M

# === LLM Query ===
def query_deepseek(prompt, api_key, model_name, max_retries=5):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"[Rate limit] Waiting {wait:.2f}s before retrying...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded maximum retries due to rate limiting.")

# === Prompt Building ===
def build_choice_prompt(sentences, phase):
    base = (
        "You are an expert annotator working on relation classification in the financial domain.\\n"
        "We are using active learning to select the most informative training samples.\\n"
    )
    specific = f"Choose exactly ONE sentence from the following {len(sentences)} that is most informative for model training.\\n"
    instruction = (
        "\\nList of sentences:\\n" +
        "\\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)]) +
        "\\n\\nRespond with ONLY the number of the best sentence (e.g., 1). DO NOT explain your answer. DO NOT include any text other than the number.\\n"
    )
    return base + specific + instruction

# === Chunk-Based Selection ===
def chunk_and_select(dataset, indices, chunk_size, api_key, model_name, phase, output_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yes_path = os.path.join(OUTPUT_DIR, f"{output_name}_yes.txt")

    # Read already completed selections
    already_selected = set()
    if os.path.exists(yes_path):
        with open(yes_path, "r") as f:
            already_selected = {int(line.strip()) for line in f.readlines()}

    selected_indices = list(already_selected)

    with open(yes_path, "a") as yes_file:
        for i in tqdm(range(0, len(indices), chunk_size), desc=f"Phase {phase} Resume Selection"):
            chunk = indices[i:i+chunk_size]
            if len(chunk) < chunk_size:
                continue

            if any(idx in already_selected for idx in chunk):
                continue  # Skip chunk if already selected from

            sentences = [dataset[idx]['sentence_with_entity_tags'] for idx in chunk]
            prompt = build_choice_prompt(sentences, phase)

            try:
                response = query_deepseek(prompt, api_key, model_name)
                response = str(response).strip()
                match = re.search(r"\\b(\\d+)\\b", response)
                if match:
                    choice = int(match.group(1))
                    if 1 <= choice <= len(chunk):
                        selected_idx = chunk[choice - 1]
                        yes_file.write(f"{selected_idx}\\n")
                        selected_indices.append(selected_idx)
                    else:
                        raise ValueError(f"Number {choice} out of range for chunk size {len(chunk)}.")
                else:
                    raise ValueError(f"No valid number in response: '{response}'")
            except Exception as e:
                print(f"[Error on chunk {i}-{i+chunk_size}] {e}")
                continue

    return selected_indices

# === Main Resume Logic ===
def main():
    dataset = load_from_disk(DATASET_DIR)
    train_data = dataset["train"]

    phase1_path = os.path.join(OUTPUT_DIR, "phase1_yes.txt")
    if not os.path.exists(phase1_path):
        raise RuntimeError("Missing phase1_yes.txt — Phase 1 must be completed first.")

    with open(phase1_path, "r") as f:
        phase1_indices = [int(line.strip()) for line in f.readlines()]

    print(f"Resuming Phase 2: using deepseek-reasoner on {len(phase1_indices)} phase 1 selections")
    chunk_and_select(
        dataset=train_data,
        indices=phase1_indices,
        chunk_size=CHUNK_SIZE_2,
        api_key=DEEPSEEK_API_KEY,
        model_name="deepseek-reasoner",
        phase=2,
        output_name="phase2"
    )

    print("✅ Phase 2 resumed and completed.")

if __name__ == "__main__":
    main()


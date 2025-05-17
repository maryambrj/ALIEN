import os
import json
import random
import requests
from datasets import load_from_disk
from tqdm import tqdm
import re
from dotenv import load_dotenv

# === Config ===
load_dotenv()  # Load from .env file
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DATASET_DIR = os.path.expanduser("~/active_learning/processed_data/refind")
OUTPUT_DIR = "./selection_outputs/two_stage_chunk"
CHUNK_SIZE_1 = 2  # N: Agent 1 selects 1 out of N
CHUNK_SIZE_2 = 5   # M: Agent 2 selects 1 out of M

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
                import time; time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded maximum retries due to rate limiting.")

# === Prompt Building ===
def build_choice_prompt(sentences, phase):
    base = (
        "You are an expert annotator working on relation classification in the financial domain.\n"
        "We are using active learning to select the most informative training samples.\n"
    )
    if phase == 1:
        specific = f"From the following {len(sentences)} options, choose exactly ONE sentence that is the most useful for improving a relation classification model.\n"
    elif phase == 2:
        specific = f"Refining prior selections: choose exactly ONE sentence from the following {len(sentences)} that is the most informative for model training.\n"

    instruction = (
        "\nList of sentences:\n" +
        "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)]) +
        "\n\nRespond with ONLY the number of the best sentence (e.g., 1). DO NOT explain your answer. DO NOT include any text other than the number.\n"
    )

    return base + specific + instruction


# === Chunk-Based Selection ===
def chunk_and_select(dataset, indices, chunk_size, api_key, model_name, phase, output_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selected_indices = []

    yes_path = os.path.join(OUTPUT_DIR, f"{output_name}_yes.txt")

    with open(yes_path, "w") as yes_file:
        for i in tqdm(range(0, len(indices), chunk_size), desc=f"Phase {phase} Chunked Selection"):
            chunk = indices[i:i+chunk_size]
            if len(chunk) < chunk_size:
                continue  # Skip incomplete final chunk

            sentences = [dataset[idx]['sentence_with_entity_tags'] for idx in chunk]
            prompt = build_choice_prompt(sentences, phase)

            try:
                response = query_deepseek(prompt, api_key, model_name)
                response = str(response).strip()
                match = re.search(r"\b(\d+)\b", response)
                if match:
                    choice = int(match.group(1))
                    if 1 <= choice <= len(chunk):
                        selected_idx = chunk[choice - 1]  # convert it to a 0-based index
                        yes_file.write(f"{selected_idx}\n")
                        selected_indices.append(selected_idx)
                    else:
                        raise ValueError(f"Number {choice} is out of range for chunk of size {len(chunk)}.")
                else:
                    raise ValueError(f"No valid number found in response: '{response}'")
            except Exception as e:
                print(f"[Error on chunk {i}-{i+chunk_size}] {e}")
                continue

    return selected_indices

# === Main ===
def main():
    dataset = load_from_disk(DATASET_DIR)
    train_data = dataset["train"]
    all_indices = list(range(len(train_data)))
    random.seed(42)
    random.shuffle(all_indices)

    print(f"Phase 1: 1 out of every {CHUNK_SIZE_1} samples using deepseek-chat")
    phase1_selected = chunk_and_select(
        train_data, all_indices, CHUNK_SIZE_1,
        api_key=DEEPSEEK_API_KEY, model_name="deepseek-chat",
        phase=1, output_name="phase1"
    )

    print(f"Phase 2: 1 out of every {CHUNK_SIZE_2} selected samples using deepseek-reasoner")
    chunk_and_select(
        train_data, phase1_selected, CHUNK_SIZE_2,
        api_key=DEEPSEEK_API_KEY, model_name="deepseek-reasoner",
        phase=2, output_name="phase2"
    )

    print("Two-stage chunked selection complete.")

if __name__ == "__main__":
    main()

import os
import json
import requests
from datasets import load_from_disk
from tqdm import tqdm

# === Configuration ===
DEEPSEEK_API_KEY = "sk-2377e5a6b92e4847b17ed9f51bdeebd9" 
DATASET_DIR = os.path.expanduser("~/active_learning/processed_data/refind")
OUTPUT_DIR = "./selection_outputs"
BUDGET_FRACTION = 0.1
MODEL_NAME = "deepseek-chat"

# === DeepSeek Call ===
def query_deepseek(prompt, api_key):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# === Agent Class ===
class ActiveLearningAgent:
    def __init__(self, llm_decision_fn, budget, out_dir):
        self.llm_decision_fn = llm_decision_fn
        self.budget = budget
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.selected_path = os.path.join(out_dir, "selected.txt")
        self.skipped_path = os.path.join(out_dir, "skipped.txt")

    def build_prompt(self, sentence):
        return (
            "You are an expert data annotator working on relation classification in the financial domain.\n"
            "We are performing active learning, so we aim to select only the most helpful training samples to label and train on.\n"
            "We want to select only around 10 percent of the available dataset, so choose carefully.\n"
            "Given the sentence below (with entity tags), decide whether including it will likely improve a classifier that will be fine-tuned on this data.\n"
            "Respond with ONLY 'YES' or 'NO'.\n\n"
            f"Sentence:\n\"{sentence}\"\n"
        )

    def run(self, dataset):
        selected_count = 0

        with open(self.selected_path, "w") as selected_file, open(self.skipped_path, "w") as skipped_file:

            for idx in tqdm(range(len(dataset)), desc="Agent Evaluating"):
                sentence = dataset[idx]["sentence_with_entity_tags"]
                prompt = self.build_prompt(sentence)
                try:
                    decision = self.llm_decision_fn(prompt).strip().lower()
                    if "yes" in decision:
                        selected_file.write(f"{idx}\n")
                        selected_count += 1
                    else:
                        skipped_file.write(f"{idx}\n")
                except Exception as e:
                    print(f"[Error on sample {idx}] {e}")
                    skipped_file.write(f"{idx}\n")
                    continue

                if selected_count >= self.budget:
                    print(f"Reached budget of {self.budget} selected samples.")
                    break

def main():
    dataset = load_from_disk(DATASET_DIR)
    train_data = dataset["train"]
    budget = int(BUDGET_FRACTION * len(train_data))

    print(f"Active Learning Agent starting: selecting {budget} of {len(train_data)} samples")

    agent = ActiveLearningAgent(
        llm_decision_fn=lambda prompt: query_deepseek(prompt, api_key=DEEPSEEK_API_KEY),
        budget=budget,
        out_dir=OUTPUT_DIR
    )

    agent.run(train_data)

    print(f"Done. Selected samples saved to {agent.selected_path}")
    print(f"Skipped samples saved to {agent.skipped_path}")

if __name__ == "__main__":
    main()

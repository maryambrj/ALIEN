import csv
import json
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing_extensions import Literal
from sklearn.metrics import accuracy_score, f1_score
from openai import OpenAI
import instructor

load_dotenv("./../.env")
API_KEY = os.getenv("OPEN_AI_API")

client = instructor.patch(OpenAI(api_key=API_KEY), mode=instructor.Mode.FUNCTIONS)
client.max_retries = 2

MODEL_NAME = 'gpt-4o'

ALL_LABELS = [
    'headquartered_in', 'formed_in', 'title', 'shares_of', 'loss_of',
    'acquired_on', 'agreement_with', 'operations_in', 'subsidiary_of',
    'employee_of', 'no_relation', 'cost_of', 'acquired_by',
    'member_of', 'profit_of', 'revenue_of', 'founder_of',
    'formed_on', 'attended'
]

FEW_SHOT_EXAMPLES = """
Examples:
Sentence: "WhatsApp was acquired by Meta."
Entity 1: WhatsApp
Entity 2: Meta
→ acquired_by

Sentence: "It was acquired on Wednesday."
Entity 1: It
Entity 2: Wednesday
→ acquired_on

Sentence: "He was the founder of Microsoft."
Entity 1: He
Entity 2: Microsoft
→ founder_of

Sentence: "Apple mentioned Amazon during the conference."
Entity 1: Apple
Entity 2: Amazon
→ no_relation
"""

class Entity(BaseModel):
    name: str
    type: str

class RelationshipClassificationInput(BaseModel):
    entity1: Entity
    entity2: Entity
    context_sentence: str

class RelationshipClassificationOutput(BaseModel):
    relationship: Literal[
        'headquartered_in', 'formed_in', 'title', 'shares_of', 'loss_of',
        'acquired_on', 'agreement_with', 'operations_in', 'subsidiary_of',
        'employee_of', 'no_relation', 'cost_of', 'acquired_by',
        'member_of', 'profit_of', 'revenue_of', 'founder_of',
        'formed_on', 'attended'
    ]
    confidence: float = Field(description='Model confidence between 0 and 1.')

def get_relationship_prediction(input_data: RelationshipClassificationInput) -> dict:
    label_list = "\n".join(ALL_LABELS)
    prompt = (
        f"Given the sentence:\n\"{input_data.context_sentence}\"\n\n"
        f"Entity 1: {input_data.entity1.name} ({input_data.entity1.type})\n"
        f"Entity 2: {input_data.entity2.name} ({input_data.entity2.type})\n\n"
        f"Classify the relationship between Entity 1 and Entity 2.\n\n"
        f"Respond with ONLY one label from the list below (copy exactly, no other words):\n"
        f"{label_list}\n\n"
        f"Here are some examples for guidance:\n"
        f"{FEW_SHOT_EXAMPLES}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_model=RelationshipClassificationOutput,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized expert in relation classification.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.model_dump()
    except Exception as e:
        print(f"API Error: {e}")
        return {"relationship": "no_relation", "confidence": 0.0}

def load_entity_story_rows_from_csv(csv_path: str):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row.get("sentence") or not row.get("head_entity_text") or not row.get("tail_entity_text"):
                continue
            rows.append(
                {
                    "entity": row["head_entity_text"].strip(),
                    "target_entity": row["tail_entity_text"].strip(),
                    "sentence": row["sentence"].strip(),
                    "label": row["relation"].strip().lower() if "relation" in row else "no_relation"
                }
            )
    return rows


CSV_PATH = "test_refined_filtered.csv"
OUTPUT_CSV = "predictions-fewshot-gpt.csv"

data = load_entity_story_rows_from_csv(CSV_PATH)
true_labels = []
predicted_labels = []

with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "sentence", "head_entity_text", "tail_entity_text", "true_label", "predicted_label", "confidence"
    ])
    writer.writeheader()

    for item in tqdm(data, desc="Processing"):
        input_obj = RelationshipClassificationInput(
            entity1=Entity(name=item["entity"], type="UNKNOWN"),
            entity2=Entity(name=item["target_entity"], type="UNKNOWN"),
            context_sentence=item["sentence"]
        )

        output = get_relationship_prediction(input_obj)

        true_labels.append(item['label'])
        predicted_labels.append(output["relationship"])

        writer.writerow({
            "sentence": item["sentence"],
            "head_entity_text": item["entity"],
            "tail_entity_text": item["target_entity"],
            "true_label": item["label"],
            "predicted_label": output["relationship"],
            "confidence": output["confidence"]
        })
        f.flush()

acc = accuracy_score(true_labels, predicted_labels)
f1_macro = f1_score(true_labels, predicted_labels, average="macro")
f1_micro = f1_score(true_labels, predicted_labels, average="micro")

print(f"Accuracy     : {acc:.4f}")
print(f"F1 Macro     : {f1_macro:.4f}")
print(f"F1 Micro     : {f1_micro:.4f}")
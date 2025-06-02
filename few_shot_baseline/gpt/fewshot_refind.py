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

load_dotenv()
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
Sentence: "In September 2014 , as part of the removal of anti - dilution , price reset and change of control provisions in various securities that had caused those securities to be classified as derivative liabilities , CorMedix Inc. entered into a Consent and Exchange Agreement with Manchester, pursuant to which Manchester had a right of 60 % participation in equity financings undertaken by CorMedix Inc."
Entity 1: CorMedix Inc.
Entity 2: Manchester
→ agreement_with

Sentence: "In the United States, NUVASIVE INC sell NUVASIVE INC products through a combination of exclusive independent sales agents and directly-employed sales personnel."
Entity 1: NUVASIVE INC
Entity 2: the United States
→ operations_in
"""

# Sentence: "Illinois EMCASCO was formed in Illinois in 1976 (and was re-domesticated to Iowa in 2001), Dakota Fire was formed in North Dakota in 1957 and EMCASCO was formed in Iowa in 1958, all for the purpose of writing property and casualty insurance."
# Entity 1: EMCASCO
# Entity 2: 1976
# → formed_on

# Sentence: "Dr. Smith also served as a member of the Industrial Associates of the School of Earth Sciences at Stanford University for several years."
# Entity 1: Smith
# Entity 2: the Industrial Associates of the School of Earth Sciences
# → member_of

# Sentence: "Mr. Schriesheim also served as a director of Dobson Communications Corp. from 2004 to 2007, a director of Lawson Software from 2006 to 2011, a director and Co-Chairman of MSC Software Corporation from 2007 to 2009 and a director of Georgia Gulf Corporation from 2009 to 2010."
# Entity 1: Schriesheim
# Entity 2: Lawson Software
# → employee_of

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


# CSV_PATH = "./datasets/refind_data/test_refined_filtered.csv"
# OUTPUT_CSV = "predictions-Refind-5shot-gemini.csv"

# data = load_entity_story_rows_from_csv(CSV_PATH)
# true_labels = []
# predicted_labels = []



# Assume your get_relationship_prediction() is defined and works

INPUT_CSV = "predictions-Refind-0shot-gpt-final.csv"
OUTPUT_CSV = "predictions-Refind-0shot-gpt-final-final.csv"

rows_to_retry = []
rows_ok = []

# 1. Read existing predictions and collect rows needing retry
with open(INPUT_CSV, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    all_rows = list(reader)

    for row in all_rows:
        if float(row["confidence"]) == 0.0:
            rows_to_retry.append(row)
        else:
            rows_ok.append(row)

print(f"Found {len(rows_to_retry)} rows to retry.")

# 2. Retry only the 0.0 confidence rows
for row in rows_to_retry:
    input_obj = RelationshipClassificationInput(
        entity1=Entity(name=row["head_entity_text"], type="UNKNOWN"),
        entity2=Entity(name=row["tail_entity_text"], type="UNKNOWN"),
        context_sentence=row["sentence"]
    )
    # You can add a retry loop here if you want to retry more than once per row
    try:
        output = get_relationship_prediction(input_obj)
        row["predicted_label"] = output["relationship"]
        row["confidence"] = output["confidence"]
        time.sleep(1)  # avoid rate limits if needed
    except Exception as e:
        print(f"Retry failed for row: {row}")
        print(e)
        # Keep as is

# 3. Write all results (ok + retried) back to a new CSV
with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as outfile:
    fieldnames = [
        "sentence", "head_entity_text", "tail_entity_text", "true_label", "predicted_label", "confidence"
    ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows_ok + rows_to_retry:
        writer.writerow(row)

print(f"Retried predictions saved to {OUTPUT_CSV}")


# with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=[
#         "sentence", "head_entity_text", "tail_entity_text", "true_label", "predicted_label", "confidence"
#     ])
#     writer.writeheader()

#     for item in tqdm(data, desc="Processing"):
#         input_obj = RelationshipClassificationInput(
#             entity1=Entity(name=item["entity"], type="UNKNOWN"),
#             entity2=Entity(name=item["target_entity"], type="UNKNOWN"),
#             context_sentence=item["sentence"]
#         )

#         output = get_relationship_prediction(input_obj)

#         true_labels.append(item['label'])
#         predicted_labels.append(output["relationship"])

#         writer.writerow({
#             "sentence": item["sentence"],
#             "head_entity_text": item["entity"],
#             "tail_entity_text": item["target_entity"],
#             "true_label": item["label"],
#             "predicted_label": output["relationship"],
#             "confidence": output["confidence"]
#         })
#         f.flush()

# acc = accuracy_score(true_labels, predicted_labels)
# f1_macro = f1_score(true_labels, predicted_labels, average="macro")
# f1_micro = f1_score(true_labels, predicted_labels, average="micro")

# print(f"Accuracy     : {acc:.4f}")
# print(f"F1 Macro     : {f1_macro:.4f}")
# print(f"F1 Micro     : {f1_micro:.4f}")
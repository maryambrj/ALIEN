import csv
import json
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing_extensions import Literal
from sklearn.metrics import accuracy_score, f1_score
# from openai import OpenAI
from google import generativeai as genai
# import instructor

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-05-20")


ALL_LABELS = [
    'headquartered_in', 'formed_in', 'title', 'shares_of', 'loss_of',
    'acquired_on', 'agreement_with', 'operations_in', 'subsidiary_of',
    'employee_of', 'no_relation', 'cost_of', 'acquired_by',
    'member_of', 'profit_of', 'revenue_of', 'founder_of',
    'formed_on', 'attended'
]

# FEW_SHOT_EXAMPLES = """
# Examples:
# Sentence: "In September 2014 , as part of the removal of anti - dilution , price reset and change of control provisions in various securities that had caused those securities to be classified as derivative liabilities , CorMedix Inc. entered into a Consent and Exchange Agreement with Manchester, pursuant to which Manchester had a right of 60 % participation in equity financings undertaken by CorMedix Inc."
# Entity 1: CorMedix Inc.
# Entity 2: Manchester
# → agreement_with

# Sentence: "In the United States, NUVASIVE INC sell NUVASIVE INC products through a combination of exclusive independent sales agents and directly-employed sales personnel."
# Entity 1: NUVASIVE INC
# Entity 2: the United States
# → operations_in
# """

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
        f"You are a specialized expert in relation classification.\n"
        f"Given the sentence:\n\"{input_data.context_sentence}\"\n\n"
        f"Entity 1: {input_data.entity1.name} ({input_data.entity1.type})\n"
        f"Entity 2: {input_data.entity2.name} ({input_data.entity2.type})\n\n"
        f"Classify the relationship between Entity 1 and Entity 2.\n\n"
        f"Respond with ONLY one label from the list below (copy exactly, no other words):\n"
        f"{label_list}\n\n"
    )
        # f"Here are some examples for guidance:\n"
        # f"{FEW_SHOT_EXAMPLES}"

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        predicted_label = result_text if result_text in ALL_LABELS else "no_relation"
        return {"relationship": predicted_label, "confidence": 1.0}
    except Exception as e:
        print(f"Gemini API Error: {e}")
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


CSV_PATH = "./datasets/refind_data/test_refined_filtered.csv"
OUTPUT_CSV = "predictions-Refind-0shot-gemini.csv"

data = load_entity_story_rows_from_csv(CSV_PATH)
true_labels = []
predicted_labels = []

processed_keys = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # You can use sentence or a combination as the unique key:
            key = (row["sentence"], row["head_entity_text"], row["tail_entity_text"])
            processed_keys.add(key)

with open(OUTPUT_CSV, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "sentence", "head_entity_text", "tail_entity_text", "true_label", "predicted_label", "confidence"
    ])
    if os.stat(OUTPUT_CSV).st_size == 0:
        writer.writeheader()

    for item in tqdm(data, desc="Processing"):
        key = (item["sentence"], item["entity"], item["target_entity"])
        if key in processed_keys:
            continue 

        input_obj = RelationshipClassificationInput(
                entity1=Entity(name=item["entity"], type="UNKNOWN"),
                entity2=Entity(name=item["target_entity"], type="UNKNOWN"),
                context_sentence=item["sentence"]
            )

        output = get_relationship_prediction(input_obj)

        true_labels.append(item['label'])
        predicted_labels.append(output["relationship"])

        confidence = output["confidence"]
        if confidence == 0.0:
            print(f"Skipping due to API failure: {key}")
        
        writer.writerow({
            "sentence": item["sentence"],
            "head_entity_text": item["entity"],
            "tail_entity_text": item["target_entity"],
            "true_label": item["label"],
            "predicted_label": output["relationship"],
            "confidence": confidence 
        })

        f.flush()

acc = accuracy_score(true_labels, predicted_labels)
f1_macro = f1_score(true_labels, predicted_labels, average="macro")
f1_micro = f1_score(true_labels, predicted_labels, average="micro")

print(f"Accuracy     : {acc:.4f}")
print(f"F1 Macro     : {f1_macro:.4f}")
print(f"F1 Micro     : {f1_micro:.4f}")
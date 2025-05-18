import os
import json
from tqdm import tqdm
import pandas as pd
import datasets
from datasets import concatenate_datasets, Dataset, DatasetDict

# dataset_dir = './raw_data/refind/'
train_data = json.load(
    open('test_refined_official.json', 'r')
)
# test_data = json.load(
#     open(dataset_dir + 'test_refind_official.json', 'r')
# )
# dev_data = json.load(
#     open(dataset_dir + 'dev_refind_official.json', 'r')
# )

label_to_idx = {
    'no_relation': 0,
    'org:date:acquired_on': 1,
    'org:date:formed_on': 2,
    'org:gpe:formed_in': 3,
    'org:gpe:headquartered_in': 4,
    'org:gpe:operations_in': 5,
    'org:money:cost_of': 6,
    'org:money:loss_of': 7,
    'org:money:profit_of': 8,
    'org:money:revenue_of': 9,
    'org:org:acquired_by': 10,
    'org:org:agreement_with': 11,
    'org:org:shares_of': 12,
    'org:org:subsidiary_of': 13,
    'pers:gov_agy:member_of': 14,
    'pers:org:employee_of': 15,
    'pers:org:founder_of': 16,
    'pers:org:member_of': 17,
    'pers:title:title': 18,
    'pers:univ:attended': 19,
    'pers:univ:employee_of': 20,
    'pers:univ:member_of': 21
}

def create_list_of_json(data_list):
    data = []
    for item in tqdm(data_list, total=len(data_list)):
        temp = {
            'id': item['id'],
            'docid': item['docid'],
            'sentence': ' '.join(item['token']),
            'head_entity_text': ' '.join(item['token'][item['e1_start']: item['e1_end']]),
            'tail_entity_text': ' '.join(item['token'][item['e2_start']: item['e2_end']]),
            'head_entity_char_idxs': [item['e1_start'], item['e1_end']],
            'tail_entity_char_idxs': [item['e2_start'], item['e2_end']],
            'relation': item['relation'],
            'label': label_to_idx[item['relation']],
            'relation_group': item['rel_group'],
            'e1_type': item['e1_type'],
            'e2_type': item['e2_type']
        }
        data.append(temp)

    return data

train_data = create_list_of_json(train_data)

# --- CSV Export block ---
df = pd.DataFrame(train_data)
csv_columns = ['head_entity_text', 'tail_entity_text', 'relation', 'sentence']

def clean_entity_text(s):
    return s.replace(',', '').replace('.', '')

# Clean up entities for other characters as you did before (if needed)
df['head_entity_text'] = df['head_entity_text'].apply(clean_entity_text)
df['tail_entity_text'] = df['tail_entity_text'].apply(clean_entity_text)

def process_relation(rel):
    if rel.count(':') >= 2:
        parts = rel.split(':')
        return ':'.join(parts[2:])
    else:
        return rel

df['relation'] = df['relation'].apply(process_relation)

# ---------- DUPE REMOVAL BASED ON SPACE-INSENSITIVE COLUMNS ----------
# Add temporary deduplication columns, stripping all whitespace for dupe-check only
for col in ['head_entity_text', 'tail_entity_text', 'relation', 'sentence']:
    df[f'dedup_{col}'] = df[col].str.replace(' ', '', regex=False)

dedup_columns = [f'dedup_{col}' for col in ['head_entity_text', 'tail_entity_text', 'relation', 'sentence']]
df = df.drop_duplicates(subset=dedup_columns, keep='first')

# Limit each unique relation value to maximum 10 rows (choose random 10 for each relation)
df = df.groupby('relation', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 5), random_state=42)).reset_index(drop=True)

# Export original columns (not the dedup temp ones)
df[csv_columns].to_csv('test_refined_filtered_lite.csv', index=False)

# Clean up temp columns for further use (if needed)
# df = df.drop(columns=dedup_columns)
# # --- end CSV Export block ---
#
# train_dataset = Dataset.from_list(train_data)
# # valid_dataset = Dataset.from_list(dev_data)
# # test_dataset = Dataset.from_list(test_data)
#
# dataset = DatasetDict(
#     {
#         'train': train_dataset,
#         # 'test': test_dataset,
#         # 'validation': valid_dataset
#     }
# )
#
# dataset.save_to_disk('./processed_data/refined')
# dataset = datasets.load_from_disk('./processed_data/refined')
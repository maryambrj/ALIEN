import json
import pandas as pd

# Read JSON
with open('test_core.json', 'r') as f:
    data = json.load(f)

# Process and extract needed fields
rows = []
for item in data:
    row = {
        'relation': item.get('relation', ''),
        'invert_relation': item.get('invert_relation', ''),
        'e1_name': item.get('e1_name', ''),
        'e2_name': item.get('e2_name', ''),
        'context': ' '.join(item.get('context', []))
    }
    rows.append(row)

# Write to CSV
df = pd.DataFrame(rows, columns=['relation', 'invert_relation', 'e1_name', 'e2_name', 'context'])
df.to_csv('test_core.csv', index=False)
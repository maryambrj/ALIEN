import os
import re

# === Config ===
PHASE1_FILE = "./selection_outputs/two_stage_chunk/phase1_yes.txt"
PHASE2_FILE = "./selection_outputs/two_stage_chunk/phase2_yes.txt"
LOG_FILE = "./slurm-56458993.out"  # Replace with your actual log output

CHUNK_SIZE = 5

# === Load Phase 1 Indices ===
with open(PHASE1_FILE, "r") as f:
    phase1_indices = [int(line.strip()) for line in f]

# === Load Already Selected Phase 2 Indices ===
existing_phase2 = set()
if os.path.exists(PHASE2_FILE):
    with open(PHASE2_FILE, "r") as f:
        existing_phase2 = {int(line.strip()) for line in f}

# === Parse Log for Recoverable Errors ===
recovered = []
pattern = re.compile(r"\[Error on chunk (\d+)-(\d+)] No valid number in response: '(\d+)'")

with open(LOG_FILE, "r") as log:
    for line in log:
        match = pattern.search(line)
        if match:
            start, end, number = map(int, match.groups())
            chunk = phase1_indices[start:end]
            if 1 <= number <= len(chunk):
                selected = chunk[number - 1]
                if selected not in existing_phase2:
                    recovered.append(selected)

# === Write Recovered Indices to Phase 2 File ===
with open(PHASE2_FILE, "a") as f:
    for idx in recovered:
        f.write(f"{idx}\n")

print(f"✅ Recovered {len(recovered)} indices and appended to {PHASE2_FILE}")

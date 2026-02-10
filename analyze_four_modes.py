
import json
from pathlib import Path

base_dir = Path("/tmp2/cctsai/project/RAG/experiment_outputs/run_20260210_172430")

modes = ["kg", "kg-contextual", "hybrid-kg", "hybrid-kg-contextual"]
files = {m: base_dir / f"eval_{m}.jsonl" for m in modes}

data = {m: {} for m in modes}

# Load data
for m, fpath in files.items():
    if not fpath.exists():
        print(f"Skipping {m}, file not found.")
        continue
    with open(fpath, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item.get('query', {}).get('query_id')
            if qid:
                data[m][qid] = item

# Find common queries
common_ids = set(data[modes[0]].keys())
for m in modes[1:]:
    common_ids &= set(data[m].keys())

print(f"Analyzing {len(common_ids)} common queries.\n")

# Select a few distinct queries to analyze
# We look for cases where KG-Ctx > KG AND Hybrid+KG > Hybrid+KG-Ctx (the user's observed pattern)
target_ids = []
for qid in common_ids:
    s_kg = data["kg"][qid].get("ROUGELScore", 0)
    s_kg_ctx = data["kg-contextual"][qid].get("ROUGELScore", 0)
    s_hyb_kg = data["hybrid-kg"][qid].get("ROUGELScore", 0)
    s_hyb_kg_ctx = data["hybrid-kg-contextual"][qid].get("ROUGELScore", 0)
    
    # Check if pattern roughly holds
    if (s_kg_ctx >= s_kg) and (s_hyb_kg > s_hyb_kg_ctx):
        target_ids.append(qid)

print(f"Found {len(target_ids)} queries matching the pattern (KG-Ctx >= KG and Hyb+KG > Hyb+KG-Ctx).")
if not target_ids:
    print("No exact matches for pattern, showing random common queries.")
    target_ids = list(common_ids)[:3]

# Limit to 2 examples
target_ids = target_ids[:2]

for qid in target_ids:
    print(f"\n{'='*80}")
    print(f"Query ID: {qid}")
    print(f"Query: {data['kg'][qid]['query']['content']}")
    print(f"{'='*80}")
    
    for m in modes:
        item = data[m][qid]
        score = item.get("ROUGELScore", 0)
        refs = item.get("prediction", {}).get("references", [])
        pred = item.get("prediction", {}).get("content", "")[:100].replace('\n', ' ')
        
        print(f"\n--- Mode: {m} (ROUGE: {score:.4f}) ---")
        print(f"prediction: {pred}...")
        print(f"Retrieved {len(refs)} chunks:")
        for i, ref in enumerate(refs[:3]): # Show top 3 chunks
            # Truncate for display
            display_ref = ref[:150].replace('\n', ' ')
            print(f"  [{i+1}] {display_ref}...")

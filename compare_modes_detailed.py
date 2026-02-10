
import json
from pathlib import Path

base_dir = Path("/tmp2/cctsai/project/RAG/experiment_outputs/run_20260210_172430")
file1 = base_dir / "eval_hybrid.jsonl"
file2 = base_dir / "eval_hybrid-kg.jsonl"

def load_data(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item.get('query', {}).get('query_id')
            if qid:
                data[qid] = item
    return data

data1 = load_data(file1)
data2 = load_data(file2)

common_ids = set(data1.keys()) & set(data2.keys())
print(f"Comparing {len(common_ids)} common queries.\n")

same_refs = 0
same_preds = 0
relevance_match = 0

for qid in common_ids:
    item1 = data1[qid]
    item2 = data2[qid]
    
    # Check References (Retrieved Chunks)
    refs1 = item1.get('prediction', {}).get('references', [])
    refs2 = item2.get('prediction', {}).get('references', [])
    # Sort to ignore order differences if any (though usually order matters in RAG)
    if refs1 == refs2:
        same_refs += 1
        
    # Check Prediction Content (Answer)
    pred1 = item1.get('prediction', {}).get('content', "").strip()
    pred2 = item2.get('prediction', {}).get('content', "").strip()
    if pred1 == pred2:
        same_preds += 1
        
    # Check Metrics
    irr1 = item1.get('irrelevance')
    irr2 = item2.get('irrelevance')
    if irr1 == irr2:
        relevance_match += 1

print(f"Identical References: {same_refs} / {len(common_ids)}")
print(f"Identical Predictions: {same_preds} / {len(common_ids)}")
print(f"Identical Irrelevance Score: {relevance_match} / {len(common_ids)}")

print("\n--- Difference Sample ---")
# Find a case where references differ but metrics might be same or similar
count = 0
for qid in common_ids:
    item1 = data1[qid]
    item2 = data2[qid]
    
    refs1 = item1.get('prediction', {}).get('references', [])
    refs2 = item2.get('prediction', {}).get('references', [])
    
    if refs1 != refs2:
        print(f"\nQuery ID: {qid}")
        print(f"Hybrid Refs Count: {len(refs1)}")
        print(f"Hybrid+KG Refs Count: {len(refs2)}")
        print(f"Metric (Irrelevance): Hybrid={item1.get('irrelevance')} vs KG={item2.get('irrelevance')}")
        
        # Show first few words of first diff reference
        print("First Hybrid Ref Start:", refs1[0][:50] if refs1 else "None")
        print("First KG Ref Start:", refs2[0][:50] if refs2 else "None")
        
        count += 1
        if count >= 3:
            break

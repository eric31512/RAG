import json

def read_first_n_items(path, n=3):
    data = []
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= n: break
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error decoding line {i} in {path}")
    except FileNotFoundError:
        print(f"File not found: {path}")
    return data

kg_path = "experiment_outputs/run_20260210_154203/output_kg.jsonl"
kg_ctx_path = "experiment_outputs/run_20260210_154203/output_kg-contextual.jsonl"

kg_data = read_first_n_items(kg_path)
kg_ctx_data = read_first_n_items(kg_ctx_path)

print(f"Loaded {len(kg_data)} KG items and {len(kg_ctx_data)} KG-Ctx items")

for i in range(min(len(kg_data), len(kg_ctx_data))):
    print(f"\n{'='*40}")
    print(f"Item {i+1}")
    print(f"{'='*40}")
    
    print(f"Query: {kg_data[i].get('query', {}).get('content', 'N/A')}")
    
    print(f"\n--- KG Context ---")
    ctx = kg_data[i].get("prediction", {}).get("references", [])
    if isinstance(ctx, list):
        for j, c in enumerate(ctx):
            print(f"[{j}] Content: {c[:100]}...")
    else:
        print(str(ctx)[:500])
        
    print(f"\n--- KG-Ctx Context ---")
    ctx_c = kg_ctx_data[i].get("prediction", {}).get("references", [])
    if isinstance(ctx_c, list):
        for j, c in enumerate(ctx_c):
            print(f"[{j}] Content: {c[:100]}...")
    else:
        print(str(ctx_c)[:500])

import json
import os
import sys
import numpy as np

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}", file=sys.stderr)
        return []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"Error parsing line: {line}\n{e}", file=sys.stderr)
    return data

def aggregate_metrics(data):
    metrics = {
        "ROUGELScore": [],
        "Sentences_Precision": [],
        "Sentences_Recall": [],
        "Words_Precision": [],
        "Words_Recall": [],
        "completeness": [],
        "hallucination": [],
        "irrelevance": []
    }
    for item in data:
        for k in metrics.keys():
            if k in item and item[k] is not None:
                metrics[k].append(item[k])
    
    avg_metrics = {}
    for k, v in metrics.items():
        if v:
            avg_metrics[k] = np.mean(v)
        else:
            avg_metrics[k] = 0.0
    return avg_metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_results.py <exp_dir>")
        sys.exit(1)
        
    exp_dir = sys.argv[1]
    run_id = os.path.basename(exp_dir.rstrip('/'))
    
    # Define files and their display names
    # Adjust names based on what you ran
    experiments = [
        {"name": "Regular KG", "file": "eval_prefix_original_kg_all.jsonl", "output": "output_prefix_original_kg_all.jsonl"},
        {"name": "Contextual KG", "file": "eval_original_contextual_kg_all.jsonl", "output": "output_original_contextual_kg_all.jsonl"}
    ]
    
    results = {}
    valid_names = []
    
    for exp in experiments:
        filepath = os.path.join(exp_dir, exp["file"])
        if os.path.exists(filepath):
            data = load_jsonl(filepath)
            if data:
                results[exp["name"]] = aggregate_metrics(data)
                valid_names.append(exp["name"])
    
    if not results:
        print("No evaluation data found items.")
        sys.exit(1)
        
    # Generate Markdown table
    metrics_list = ["ROUGELScore", "Sentences_Precision", "Sentences_Recall", "Words_Precision", "Words_Recall", "completeness", "hallucination", "irrelevance"]
    
    md = []
    md.append("# Multi-Mode RAG Experiment Report\n")
    md.append(f"**Run ID**: {run_id.replace('run_', '')}\n")
    md.append("**Language**: en\n")
    md.append("## Results Summary\n")
    
    # Table header
    header = "| Metric | " + " | ".join(valid_names) + " |"
    divider = "|--------|" + "|".join(["-------"] * len(valid_names)) + "|"
    md.append(header)
    md.append(divider)
    
    # Table rows
    for m in metrics_list:
        row = f"| {m} | "
        values = []
        for name in valid_names:
            val = results[name].get(m, 0.0)
            values.append(f"{val:.4f}")
        row += " | ".join(values) + " |"
        md.append(row)
        
    md.append("\n## Mode Descriptions\n")
    md.append("| Mode | Description |")
    md.append("|------|-------------|")
    md.append("| **Regular KG** | KG retrieval based on `prefix_original_kg_all` |")
    md.append("| **Contextual KG** | KG retrieval based on `original_contextual_kg_all` |")
    
    md.append("\n## Files\n")
    for exp in experiments:
        if exp["name"] in valid_names:
            md.append(f"- `{exp['output']}`: RAG predictions")
            md.append(f"- `{exp['file']}`: Evaluation results")
            
    report_content = "\n".join(md)
    
    output_path = os.path.join(exp_dir, "comparison_report.md")
    with open(output_path, 'w') as f:
        f.write(report_content)
        
    print(f"Report generated successfully at: {output_path}")
    print("\n" + report_content)

if __name__ == "__main__":
    main()

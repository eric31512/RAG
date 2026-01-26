# My_RAG/sample_queries.py
import json
import random
import argparse

def sample_queries(input_file, output_file, language, n=10):
    """從 input_file 隨機抽取 n 個對應語言的 queries"""
    with open(input_file, 'r', encoding='utf-8') as f:
        all_queries = [json.loads(line) for line in f]
    
    # 過濾語言
    lang_queries = [q for q in all_queries if q.get('language') == language]
    
    # 隨機抽樣
    sampled = random.sample(lang_queries, min(n, len(lang_queries)))
    
    # 寫入輸出
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in sampled:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    print(f"Sampled {len(sampled)} queries for language '{language}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()
    sample_queries(args.input, args.output, args.language, args.n)
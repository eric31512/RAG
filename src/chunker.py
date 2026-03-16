from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ollama import Client
from utils import load_ollama_config
import os
import json

def _get_sliding_window(doc_text, chunk_text, window_size=6000):
    """Get a sliding window of surrounding context centered on the chunk."""
    chunk_idx = doc_text.find(chunk_text)
    if chunk_idx != -1:
        remaining = max(0, window_size - len(chunk_text))
        start = max(0, chunk_idx - remaining // 2)
        end = min(len(doc_text), chunk_idx + len(chunk_text) + remaining - remaining // 2)
        return doc_text[start:end]
    return doc_text[:window_size]

def _extract_subject_name(metadata):
    """Extract the primary subject name from metadata."""
    if not metadata:
        return ""
    for key in ("company_name", "hospital_patient_name", "court_name"):
        if key in metadata:
            return metadata[key]
    return ""

def _generate_chunk_context(language, doc_text, chunk_text, metadata=None):
    """
    Generate contextual description for a specific chunk using Ollama.
    This follows Anthropic's Contextual Retrieval approach.
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    doc_text_truncated = _get_sliding_window(doc_text, chunk_text)
    subject_name = _extract_subject_name(metadata)
    
    if language == "zh":
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\n重要：本文档的主体是「{subject_name}」。"
        
        prompt = f"""<document>
{doc_text_truncated}
</document>

以下是我们想要在整个文档中定位的块：
<chunk>
{chunk_text}
</chunk>

任务：为这个块生成一个上下文描述（50-80字），说明这段内容在整个文档中的位置和作用。{subject_instruction}

要求：
- 必须用简体中文回答
- 必须说明这段内容位于文档的哪个部分（如：文档开头、第二部分、结尾部分、财务指标部分等）
- 必须包含主体名称（公司名称/法院名称/医院名称）
- 如果是法律文档，必须包含法院名称和被告人/当事人姓名
- 如果是病历文档，必须包含医院名称和患者姓名
- 禁止直接复制原文内容
- 禁止使用代词如"该公司"、"该患者"、"本文档"等

格式示例：「这段内容位于[文档位置]，描述了[主体名称]的[主要内容]。」

请直接输出上下文描述："""
    else:
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\nIMPORTANT: The subject of this document is \"{subject_name}\"."
        
        prompt = f"""<document>
{doc_text_truncated}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Task: Generate a short context (30-50 words) describing WHERE this chunk is located in the document and its role.{subject_instruction}

Requirements:
- You MUST respond in English only
- MUST specify the location in the document (e.g., "at the beginning", "in the financial section", "at the end")
- MUST include the explicit subject name (company name/court name/hospital name)
- For legal documents, include both court name and defendant/party names
- For medical records, include both hospital name and patient name
- DO NOT copy the original text directly
- DO NOT use pronouns like "the company", "this document", "it", etc.

Format example: "This section, located in [document position], describes [subject name]'s [main content]."

Output ONLY the context description:"""
    
    try:
        response = client.generate(
            model=ollama_config["model"],
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
            }
        )
        context = response["response"].strip()
        return context
    except Exception as e:
        print(f"Error generating chunk context: {e}")
        return ""


def _rewrite_chunk_with_subjects(language, doc_text, chunk_text, metadata=None):
    """
    Rewrite a chunk in-place by replacing all pronouns and ambiguous references
    with explicit entity names, using surrounding context for resolution.
    
    Unlike _generate_chunk_context() which prepends a prefix, this function
    returns a rewritten version of the original chunk text — optimized for
    Knowledge Graph entity extraction (avoids creating Hub nodes).
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    doc_text_truncated = _get_sliding_window(doc_text, chunk_text)
    subject_name = _extract_subject_name(metadata)
    
    if language == "zh":
        subject_hint = ""
        if subject_name:
            subject_hint = f"\n提示：本文档的主体是「{subject_name}」。"
        
        prompt = f"""<surrounding_context>
{doc_text_truncated}
</surrounding_context>

请改写以下文本块，将所有代词和模糊指代替换为明确的实体名称：
<chunk>
{chunk_text}
</chunk>
{subject_hint}
规则：
- 将所有代词（他、她、它、其、该公司、该患者、本院等）替换为具体的名称
- 严禁添加原文中不存在的新信息或新句子
- 严禁删除任何原有内容
- 保持原文的句子结构和语序不变
- 仅输出改写后的文本，不要输出任何解释

改写后的文本："""
    else:
        subject_hint = ""
        if subject_name:
            subject_hint = f"\nHint: The main subject of this document is \"{subject_name}\"."
        
        prompt = f"""<surrounding_context>
{doc_text_truncated}
</surrounding_context>

Rewrite the following text chunk by replacing ALL pronouns and ambiguous references with their explicit entity names:
<chunk>
{chunk_text}
</chunk>
{subject_hint}
Rules:
- Replace every pronoun (he, she, it, they, the company, the patient, the court, etc.) with the specific name
- Do NOT add any new information or sentences that are not in the original chunk
- Do NOT remove any existing content
- Keep the original sentence structure and word order intact
- Output ONLY the rewritten text, nothing else

Rewritten text:"""
    
    try:
        response = client.generate(
            model=ollama_config["model"],
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
            }
        )
        rewritten = response["response"].strip()
        
        # Safety check: if LLM returned something way too short or too long,
        # fall back to original chunk
        if len(rewritten) < len(chunk_text) * 0.5 or len(rewritten) > len(chunk_text) * 2.0:
            print(f"Warning: Rewrite length mismatch (orig={len(chunk_text)}, rewritten={len(rewritten)}). Using original.")
            return chunk_text
        
        return rewritten
    except Exception as e:
        print(f"Error rewriting chunk: {e}")
        return chunk_text


#test later
def main_summarize(language, doc_text, metadata=None):
    """
    Generate a high-level summary for the entire document using Ollama.
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    # Truncate to avoid token limits for the document (e.g., keep first 10000 chars)
    doc_text_truncated = doc_text[:10000]
    
    # Extract subject name from metadata if available
    subject_name = _extract_subject_name(metadata)

    if language == "zh":
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\n重要：本文档涉及的核心主体是「{subject_name}」。"

        prompt = f"""<document>
            {doc_text_truncated}
        </document>

        任务：为以上文档生成一个高度概括的摘要（50-80字），准确说明这份文档的主要内容与核心目的。{subject_instruction}

        要求：
        - 必须用简体中文回答
        - 必须包含明确的主体名称（如公司名称、法院名称、医院名称等）
        - 如果是法律类文档，必须说明涉及的法院与当事人/被告人姓名
        - 如果是医疗类文档，必须说明涉及的医院与患者姓名
        - 禁止直接复制文档内的长句
        - 禁止使用模糊代词（如"该公司"、"该患者"、"此文档"等），请直接写出名称

        请直接输出摘要内容，不要包含任何额外解释："""
    else:
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\nIMPORTANT: The main subject of this document is \"{subject_name}\"."

        prompt = f"""<document>
            {doc_text_truncated}
        </document>

        Task: Generate a concise, high-level summary (30-50 words) that accurately describes the main content and purpose of this document.{subject_instruction}

        Requirements:
        - You MUST respond in English only
        - MUST explicitly include the specific subject names (e.g., company name, court name, hospital name)
        - For legal documents, explicitly state the court name and the involved parties/defendant names
        - For medical records, explicitly state the hospital name and the patient's name
        - DO NOT simply copy/paste long sentences from the text
        - DO NOT use vague pronouns (like "the company", "the patient", "this document", "it"), use actual names instead

        Output ONLY the summary text, without any additional explanations or introductory phrases:"""

    try:
        response = client.generate(
            model=ollama_config["model"],
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_ctx": 16384,  # Larger context for full document summarization
            }
        )
        summary = response["response"].strip()
        return summary
    except Exception as e:
        print(f"Error generating document summary: {e}")
        return ""

def recursive_chunk(docs, language, chunk_size, mode="contextual", num_workers=2):
    """Split documents into chunks using recursive character splitting.
    
    Args:
        docs: List of document dicts with 'content' and 'language' keys
        language: Target language ('en' or 'zh')
        chunk_size: Size of each chunk in characters
        mode: Processing mode:
            - 'contextual': prepend contextual prefix (current behavior, good for Hybrid)
            - 'subject_replace': rewrite pronouns in-place (good for KG building)
            - 'plain': no LLM processing, raw chunks only
        num_workers: Number of parallel threads for LLM calls (default=2, keep low for memory)
    """
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_size // 5}, Mode: {mode}, Workers: {num_workers}")

    # Build cache path based on mode
    if mode == "contextual":
        cache_path = f"./cache/chunk_cache/{language}_contextual_chunksize{chunk_size}"
    elif mode == "subject_replace":
        cache_path = f"./cache/chunk_cache/{language}_subject_replace_chunksize{chunk_size}"
    else:
        cache_path = f"./cache/chunk_cache/{language}_chunksize{chunk_size}"
    
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Chunk cache hit: {cache_path}")
        return chunks

    if language == "en":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
            length_function=len,
            is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", "、", "：", " ", ""])
    
    chunks = []
    
    # Phase 1: Split all documents into raw chunks (fast, CPU-only)
    raw_tasks = []  # List of (text_chunk, original_text, meta)
    for doc in docs:
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            original_text = doc['content']
            lang = doc['language']

            if lang == language:
                meta = doc.copy()
                meta.pop("content", None)
                
                try:
                    split_texts = text_splitter.split_text(original_text)
                    for text_chunk in split_texts:
                        if text_chunk.strip():
                            raw_tasks.append((text_chunk, original_text, meta))
                except Exception as e:
                    print(f"Error splitting doc: {e}")
    
    print(f"Split into {len(raw_tasks)} raw chunks. Starting LLM processing (mode={mode})...")
    
    # Phase 2: Process chunks with LLM (parallel)
    if mode == "plain":
        # No LLM processing needed
        for text_chunk, original_text, meta in raw_tasks:
            chunk_meta = meta.copy()
            chunk_meta["rewrite_mode"] = "plain"
            chunk_meta["original_content"] = text_chunk
            chunks.append({
                "page_content": text_chunk,
                "metadata": chunk_meta,
            })
    else:
        def _process_one(args):
            idx, text_chunk, original_text, meta = args
            if mode == "contextual":
                prefix = _generate_chunk_context(language, original_text, text_chunk, metadata=meta)
                if prefix:
                    page_content = f"{prefix}\n\n{text_chunk}"
                else:
                    page_content = text_chunk
                return idx, page_content, prefix, text_chunk
            elif mode == "subject_replace":
                rewritten = _rewrite_chunk_with_subjects(language, original_text, text_chunk, metadata=meta)
                return idx, rewritten, "", text_chunk
        
        # Build indexed task list to preserve order
        indexed_tasks = [(i, tc, ot, m) for i, (tc, ot, m) in enumerate(raw_tasks)]
        results = [None] * len(indexed_tasks)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one, task): task[0] for task in indexed_tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"LLM Processing ({mode})"):
                try:
                    idx, page_content, prefix, original_chunk = future.result()
                    results[idx] = (page_content, prefix, original_chunk)
                except Exception as e:
                    orig_idx = futures[future]
                    print(f"Error processing chunk {orig_idx}: {e}")
                    # Fallback to raw chunk
                    text_chunk = raw_tasks[orig_idx][0]
                    results[orig_idx] = (text_chunk, "", text_chunk)
        
        # Assemble chunks in original order
        for i, (page_content, prefix, original_chunk) in enumerate(results):
            meta = raw_tasks[i][2]
            chunk_meta = meta.copy()
            chunk_meta["rewrite_mode"] = mode
            chunk_meta["original_content"] = original_chunk
            if mode == "contextual":
                chunk_meta["contextual_prefix"] = prefix
            chunks.append({
                "page_content": page_content,
                "metadata": chunk_meta,
            })
    
    print(f"Created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    import argparse
    from utils import load_jsonl

    parser = argparse.ArgumentParser(description="Run recursive chunking with optional LLM processing.")
    parser.add_argument("--docs_path", type=str, required=True, help="Path to the documents JSONL file")
    parser.add_argument("--language", type=str, required=True, choices=["zh", "en"], help="Language to filter (zh or en)")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size (default: 128 for zh, 512 for en)")
    parser.add_argument("--mode", type=str, default="contextual", choices=["contextual", "subject_replace", "plain"],
                        help="Processing mode: contextual (prefix), subject_replace (rewrite pronouns), plain (no LLM)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel LLM workers (default: 2, keep low for <20GB memory)")
    args = parser.parse_args()

    # Default chunk size by language
    chunk_size = args.chunk_size if args.chunk_size else (128 if args.language == "zh" else 512)

    # Load documents
    docs_for_chunking = load_jsonl(args.docs_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")

    # Run recursive chunking
    chunks = recursive_chunk(docs_for_chunking, args.language, chunk_size,
                             mode=args.mode, num_workers=args.workers)

    # Save to chunk_cache (cache path is determined by mode inside recursive_chunk)
    if args.mode == "contextual":
        cache_path = f"./chunk_cache/{args.language}_contextual_chunksize{chunk_size}"
    elif args.mode == "subject_replace":
        cache_path = f"./chunk_cache/{args.language}_subject_replace_chunksize{chunk_size}"
    else:
        cache_path = f"./chunk_cache/{args.language}_chunksize{chunk_size}"
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {cache_path}")
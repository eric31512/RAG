from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from ollama import Client
from utils import load_ollama_config
import os
import json

def _generate_chunk_context(language, doc_text, chunk_text, metadata=None):
    """
    Generate contextual description for a specific chunk using Ollama.
    This follows Anthropic's Contextual Retrieval approach.
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    # Sliding window: context window centered on chunk_text (~6000 chars total)
    window_size = 6000
    chunk_idx = doc_text.find(chunk_text)
    if chunk_idx != -1:
        remaining = max(0, window_size - len(chunk_text))
        start = max(0, chunk_idx - remaining // 2)
        end = min(len(doc_text), chunk_idx + len(chunk_text) + remaining - remaining // 2)
        doc_text_truncated = doc_text[start:end]
    else:
        doc_text_truncated = doc_text[:window_size]
    
    # Extract subject name from metadata
    subject_name = ""
    if metadata:
        if "company_name" in metadata:
            subject_name = metadata["company_name"]
        elif "hospital_patient_name" in metadata:
            subject_name = metadata["hospital_patient_name"]
        elif "court_name" in metadata:
            subject_name = metadata["court_name"]
    
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
            model=ollama_config["model"],  # Uses your configured model
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
    subject_name = ""
    if metadata:
        if "company_name" in metadata:
            subject_name = metadata["company_name"]
        elif "hospital_patient_name" in metadata:
            subject_name = metadata["hospital_patient_name"]
        elif "court_name" in metadata:
            subject_name = metadata["court_name"]

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

def recursive_chunk(docs, language, chunk_size):
    """Split documents into chunks using recursive character splitting."""
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_size // 5}")

    # Build cache path
    cache_path = f"./chunk_cache/{language}_contextual_chunksize{chunk_size}"
    
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
    
    for doc in tqdm(docs, desc="Recursive Chunking"):   
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            original_text = doc['content']
            lang = doc['language']

            if lang == language:
                # Generate contextual summary for the document
                meta = doc.copy()
                meta.pop("content", None)
                
                try:
                    split_texts = text_splitter.split_text(original_text)
                    for text_chunk in split_texts:
                        if text_chunk.strip():
                            # Generate contextual prefix using Ollama
                            contextual_prefix = _generate_chunk_context(
                                language, original_text, text_chunk, metadata=meta
                            )
                            # Combine prefix + original text as page_content
                            if contextual_prefix:
                                page_content = f"{contextual_prefix}\n\n{text_chunk}"
                            else:
                                page_content = text_chunk
                            
                            chunk_meta = meta.copy()
                            chunk_meta["contextual_prefix"] = contextual_prefix
                            chunk_meta["original_content"] = text_chunk
                            
                            chunks.append({
                                "page_content": page_content,
                                "metadata": chunk_meta,
                            })
                except Exception as e:
                    print(f"Error chunking doc: {e}")
    
    print(f"Created {len(chunks)} chunks")
    return chunks
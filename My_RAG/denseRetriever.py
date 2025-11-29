import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from utils import rrf_fusion, SimpleHit
import ollama
import yaml
from pathlib import Path
import torch

def load_ollama_config(language) -> dict:
    # Try to find config in parent directories
    current_path = Path(__file__).resolve()
    
    # We'll look up to 3 levels up for a 'configs' directory
    configs_folder = None
    for i in range(4):
        candidate = current_path.parents[i] / "configs"
        if candidate.exists():
            configs_folder = candidate
            break
            
    if not configs_folder:
         # Fallback to hardcoded path if relative search fails
         configs_folder = Path(__file__).parent.parent / "configs"

    # Priority: config_local.yaml > config_submit.yaml
    config_files = ["config_local.yaml", "config_submit.yaml"]
    
    config_data = None
    for fname in config_files:
        fpath = configs_folder / fname
        if fpath.exists():
            try:
                with open(fpath, "r") as file:
                    config_data = yaml.safe_load(file)
                print(f"Loaded config from {fpath}")
                break
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue
    
    if not config_data:
        return None

    if language == "en":
        return config_data.get("EN")
    elif language == "zh":
        return config_data.get("ZH")
    
    return None

class DenseRetriever:
    def __init__(self, chunks, language="en", use_ollama=False, ollama_config=None):
        self.chunks = chunks
        self.language = language
        self.use_ollama = use_ollama
        self.ollama_config = ollama_config
        self.corpus = [chunk['page_content'] for chunk in chunks]
        

        # Dense Index
        if self.use_ollama and self.ollama_config:
            print(f"Using Ollama model: {ollama_config['model']} at {ollama_config['host']}")
            self.client = ollama.Client(host=ollama_config['host'])
            self.model_name = ollama_config['model']
            
            # Generate embeddings using Ollama
            # Note: Ollama python client might not support batch embedding efficiently in all versions, 
            # but let's try to iterate if needed or use batch if supported.
            # The denseRetriever.py uses self.client.embeddings(model=..., prompt=...) which is single input.
            # We need to loop for corpus.
            print("Generating embeddings with Ollama...")
            embeddings_list = []
            for doc in self.corpus:
                response = self.client.embeddings(model=self.model_name, prompt=doc)
                embeddings_list.append(response['embedding'])
            embeddings = np.array(embeddings_list).astype('float32')
        else:
            if language == "zh":
                model_name = 'Qwen/Qwen3-Embedding-0.6B'
            else:
                model_name = 'google/embeddinggemma-300m'
            self.encoder = SentenceTransformer(model_name)
            embeddings = self.encoder.encode(self.corpus, convert_to_numpy=True, show_progress_bar=True)
            
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=5):
        # 2. Dense Retrieval (Embedding + FAISS)
        # Generate query embedding
        if self.use_ollama:
            response = self.client.embeddings(model=self.model_name, prompt=query)
            query_vector = np.array([response['embedding']]).astype('float32')
        else:
            query_vector = self.encoder.encode([query], convert_to_numpy=True)
            
        # Normalize query vector for Cosine Similarity (Inner Product on normalized vectors)
        faiss.normalize_L2(query_vector)
        
        # Search in FAISS index
        dense_scores, dense_indices = self.index.search(query_vector, top_k)
        
        # Create SimpleHit objects for RRF fusion
        dense_hits = []
        for score, idx in zip(dense_scores[0], dense_indices[0]):
            if idx != -1:   
                dense_hits.append(SimpleHit(docid=idx, score=score))

        
        # 4. Format Results
        top_chunks = []
        for hit in dense_hits:
            docid = hit.docid
            score = hit.score
            # Retrieve the original chunk using the integer index
            # Use .copy() to avoid modifying the original chunk in memory
            original_chunk = self.chunks[docid].copy()
            original_chunk['metadata'] = original_chunk['metadata'].copy()
            
            # Inject RRF score into metadata for downstream usage
            original_chunk['metadata']['score'] = score
            top_chunks.append(original_chunk)

        return top_chunks

def create_retriever(chunks, language):
    """Creates a hybrid retriever from document chunks."""
    try:
        ollama_config = load_ollama_config(language=language)
        use_ollama = False
        if ollama_config:
             # Check if we should use ollama based on some logic or just if config exists?
             # The user request implies we want to use it if available/configured.
             # In denseRetriever.py, it falls back if config fails.
             # Here we can default to True if config is found.
             use_ollama = True
    except Exception as e:
        print(f"Failed to load Ollama config: {e}. Fallback to SentenceTransformer.")
        use_ollama = False
        ollama_config = None

    return DenseRetriever(chunks, language, use_ollama=use_ollama, ollama_config=ollama_config)

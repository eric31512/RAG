import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from ollama import Client
from generator import load_ollama_config
from sentence_transformers import CrossEncoder
import torch

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

class HybridRetriever:
    """
    Hybrid retriever using:
    - BM25 (sparse)
    - Ollama embedding (dense)
    - CrossEncoder reranker
    """

    def __init__(
        self,
        chunks,
        language: str = "en",
        reranker_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk["page_content"] for chunk in chunks]

        # --- 1. BM25 Setup (same as before) ---
        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # --- 2. Dense Embedding Setup (Ollama) ---
        self.embedding_model_name = EMBEDDING_MODEL
        ollama_cfg = load_ollama_config()
        host = ollama_cfg["host"]
        print(f"Using Ollama Embedding Model: {self.embedding_model_name} at {host}")
        self.ollama_client = Client(host=host)

        doc_embeddings = []
        for text in self.corpus:
            res = self.ollama_client.embeddings(
                model=self.embedding_model_name,
                prompt=text,
            )
            doc_embeddings.append(np.array(res["embedding"], dtype=np.float32))
        # Shape: (num_docs, dim)
        self.doc_embeddings = np.vstack(doc_embeddings)

        # --- 3. Reranker Setup (same idea, safer device selection) ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Reranker Model: {reranker_name} on device: {device}")
        self.reranker = CrossEncoder(
            reranker_name,
            max_length=512,
            device=device,
        )

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed query using Ollama embedding model.
        """
        res = self.ollama_client.embeddings(
            model=self.embedding_model_name,
            prompt=query,
        )
        return np.array(res["embedding"], dtype=np.float32)

    def _compute_dense_scores(self, query: str) -> np.ndarray:
        """
        Compute cosine similarity scores between query and all documents.
        """
        query_vec = self._embed_query(query)  # shape (dim,)
        doc_matrix = self.doc_embeddings  # shape (N, dim)

        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = doc_norms * query_norm
        denom[denom == 0] = 1e-10  # avoid division by zero

        dense_scores = (doc_matrix @ query_vec) / denom
        return dense_scores

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5):
        # --- Stage 1: Hybrid Retrieve ---
        initial_k = top_k * 10

        # BM25 scores
        if self.language == "zh":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.split(" ")
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (
                bm25_scores.max() - bm25_scores.min()
            )

        # Dense scores from Ollama embedding
        dense_scores = self._compute_dense_scores(query)

        # Normalize dense scores to [0, 1]
        if dense_scores.max() - dense_scores.min() > 1e-6:
            dense_scores = (dense_scores - dense_scores.min()) / (
                dense_scores.max() - dense_scores.min()
            )
        else:
            dense_scores = np.zeros_like(dense_scores)

        # Hybrid score
        hybrid_scores = (1 - alpha) * bm25_scores + alpha * dense_scores

        # Top initial_k candidates
        top_indices = np.argsort(hybrid_scores)[-initial_k:][::-1]

        # --- Stage 2: Rerank  ---
        candidate_pairs = []
        candidate_chunks = []

        for idx in top_indices:
            doc_text = self.chunks[idx]["page_content"]
            candidate_pairs.append([query, doc_text])
            candidate_chunks.append(self.chunks[idx])

        if len(candidate_pairs) == 0:
            return []

        rerank_scores = self.reranker.predict(candidate_pairs)
        sorted_indices_rerank = np.argsort(rerank_scores)[::-1]

        final_top_chunks = []
        for idx in sorted_indices_rerank[:top_k]:
            final_top_chunks.append(candidate_chunks[idx])

        return final_top_chunks


def create_retriever(chunks, language: str = "en"):
    """
    Factory function, interface kept the same for main.py.
    """
    return HybridRetriever(chunks, language=language)

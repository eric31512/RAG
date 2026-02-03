"""
Hybrid Retriever using LlamaIndex with Ollama embeddings (fully offline).
Combines BM25 sparse retrieval with Vector dense retrieval (RRF fusion).
Includes SimilarityPostprocessor for contextual compression.
Optionally integrates Knowledge Graph retrieval via nano-graphrag.
"""
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from pyserini_bm25 import PyseriniBM25Retriever
from utils import load_ollama_config
from reranker import Reranker
import os

# Disable OpenAI defaults - use Ollama only (fully offline)
Settings.llm = None
Settings.embed_model = None

class Retriever:
    def __init__(self, chunks, language="en", chunksize=1024, similarity_threshold=0.5, use_kg=False):
        self.language = language
        
        # Convert to LlamaIndex nodes
        nodes = [
            TextNode(
                text=c['page_content'],
                metadata={**c.get('metadata', {}), 'chunk_id': i}
            ) for i, c in enumerate(chunks)
        ]
        
        # Select embedding model (Ollama - offline)
        model = "embeddinggemma:300m" if language == "en" else "qwen3-embedding:0.6b"
        self.embed_model = OllamaEmbedding(
            model_name=model,
            base_url=load_ollama_config()['host'],
            embed_batch_size=64,
            options={"num_parallel": 6}
        )
        Settings.embed_model = self.embed_model
        
        self.retrieve_topk = 100
        # 1. BM25 Retriever
        
        if isinstance(chunksize, (int, float)) or str(chunksize).replace('.', '').isdigit():
            bm25_index_path = f"./bm25_index_cache/{language}_chunksize{chunksize}"
        else:
            bm25_index_path = f"./bm25_index_cache/{language}_{chunksize}"
        bm25 = PyseriniBM25Retriever.from_defaults(
            nodes=nodes,
            language=language,
            similarity_top_k=self.retrieve_topk,
            index_path=bm25_index_path,
            k1=1.2,
            b=0.75,
        )
        
        # 2. Vector Index with caching
        if isinstance(chunksize, (int, float)) or str(chunksize).replace('.', '').isdigit():
            vector_cache_path = f"./vector_index_cache/{language}_chunksize{chunksize}"
        else:
            vector_cache_path = f"./vector_index_cache/{language}_{chunksize}"
        
        if os.path.exists(vector_cache_path):
            # Load from cache
            print(f"Loading vector index from cache: {vector_cache_path}")
            storage_context = StorageContext.from_defaults(persist_dir=vector_cache_path)
            vector_index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        else:
            # Build and save
            print(f"Building vector index and saving to: {vector_cache_path}")
            vector_index = VectorStoreIndex(nodes, embed_model=self.embed_model, show_progress=True)
            os.makedirs(vector_cache_path, exist_ok=True)
            vector_index.storage_context.persist(persist_dir=vector_cache_path)
        
        vector = vector_index.as_retriever(similarity_top_k=self.retrieve_topk)
        if language == "zh":
            bm25_weight = 0.6
            vector_weight = 0.4
        else:
            bm25_weight = 0.5
            vector_weight = 0.5
        # 2. Hybrid Fusion (RRF)
        self.retriever = QueryFusionRetriever(
            retrievers=[bm25, vector],
            retriever_weights=[bm25_weight, vector_weight],
            similarity_top_k=self.retrieve_topk,
            num_queries=1,
            mode="relative_score",
        )
        # Cross-encoder rerank
        self.reranker_module = Reranker(
            top_n=self.retrieve_topk
        )
        
        # KG Retriever (optional)
        self.use_kg = use_kg
        self.kg_retriever = None
        if use_kg:
            try:
                from kg_retriever import create_kg_retriever
                self.kg_retriever = create_kg_retriever(language)
                print(f"[Retriever] KG retrieval enabled for {language}")
            except Exception as e:
                print(f"[Retriever] Warning: KG retrieval disabled - {e}")
                self.use_kg = False
        
    def retrieve(self, query, top_k=None, include_kg=True):
        if top_k is None:
            if self.language == "zh":
                top_k = 3
            else:
                top_k = 5
        
        # Get initial results from hybrid retriever
        init_nodes = self.retriever.retrieve(query)
        nodes = init_nodes[:20]
        final_nodes = self.reranker_module.rerank(nodes, query)
        final_nodes = final_nodes[:top_k]
        
        results = [
            {
                'page_content': n.node.get_content(),
                'metadata': {
                    'score': n.score,
                    'type': 'hybrid_retrieval',
                    **n.node.metadata
                }
            } for i, n in enumerate(final_nodes)
        ]
        
        # Add KG context if enabled
        if self.use_kg and self.kg_retriever and include_kg:
            try:
                kg_result = self.kg_retriever.retrieve_local(query)
                if kg_result.get("kg_context"):
                    results.append({
                        'page_content': kg_result["kg_context"],
                        'metadata': {
                            'score': 1.0,
                            'type': 'knowledge_graph',
                            'mode': kg_result.get("mode", "local")
                        }
                    })
                    print(f"[Retriever] Added KG context")
            except Exception as e:
                print(f"[Retriever] KG retrieval failed: {e}")
        
        return results


def create_retriever(chunks, language, chunksize, similarity_threshold=0.5, use_kg=False):
    return Retriever(chunks, language, chunksize, similarity_threshold, use_kg=use_kg)

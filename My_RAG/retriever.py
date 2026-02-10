"""
Hybrid Retriever using LlamaIndex with Ollama embeddings (fully offline).
Combines BM25 sparse retrieval with Vector dense retrieval (RRF fusion).
Includes SimilarityPostprocessor for contextual compression.
Optionally integrates Knowledge Graph retrieval via nano-graphrag.
"""
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, NodeWithScore
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
    def __init__(self, chunks, language="en", chunksize=1024, similarity_threshold=0.5, use_kg=False, contextual_kg=False):
        self.language = language
        
        # Convert to LlamaIndex nodes
        nodes = [
            TextNode(
                text=c['page_content'],
                metadata={**c.get('metadata', {}), 'chunk_id': i}
            ) for i, c in enumerate(chunks)
        ]
        
        # Select embedding model
        # Using Ollama embedding models (lightweight, shared server)
        if language == "en":
            # nomic-embed-text: 274MB, 768 dim (same as EmbeddingGemma-300m)
            self.embed_model = OllamaEmbedding(
                model_name="nomic-embed-text",
                base_url=load_ollama_config()['host'],
                embed_batch_size=64,
            )
        else:
            # Use Ollama for Chinese
            self.embed_model = OllamaEmbedding(
                model_name="qwen3-embedding:0.6b",
                base_url=load_ollama_config()['host'],
                embed_batch_size=64,
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
        self.contextual_kg = contextual_kg
        self.kg_retriever = None
        if use_kg:
            try:
                from kg_retriever import create_kg_retriever
                self.kg_retriever = create_kg_retriever(language, contextual=contextual_kg)
                kg_type = "contextual" if contextual_kg else "regular"
                print(f"[Retriever] KG retrieval enabled for {language} ({kg_type})")
            except Exception as e:
                print(f"[Retriever] Warning: KG retrieval disabled - {e}")
                self.use_kg = False
        
    def retrieve(self, query, top_k=None, mode="hybrid"):
        """
        Retrieve relevant chunks based on mode.
        
        Args:
            query: The query string
            top_k: Number of results to return
            mode: Retrieval mode - 'hybrid', 'kg', 'kg-contextual',
                  'hybrid-kg', 'hybrid-kg-contextual'
        
        Returns:
            List of retrieved chunks with metadata
        """
        if top_k is None:
            if self.language == "zh":
                top_k = 3
            else:
                top_k = 5
        
        results = []
        kg_entity_names = []
        
        # --- Step 1: KG retrieval (do this first to get entity names for query expansion) ---
        if mode in ["kg", "kg-contextual", "hybrid-kg", "hybrid-kg-contextual"]:
            if self.kg_retriever:
                try:
                    kg_result = self.kg_retriever.retrieve_local(query)
                    
                    # Extract entity names for query expansion
                    kg_entity_names = kg_result.get("entity_names", [])
                    
                    # Add KG source text chunks as separate document fragments
                    for i, chunk in enumerate(kg_result.get("source_chunks", [])):
                        results.append({
                            'page_content': chunk["content"],
                            'metadata': {
                                'score': 0.9 - (i * 0.05),  # Decreasing score
                                'type': 'kg_source_text',
                                'mode': kg_result.get("mode", "local")
                            }
                        })
                    
                    # Add compact entity/relationship summary as structured context
                    kg_context = kg_result.get("kg_context", "")
                    if kg_context:
                        results.append({
                            'page_content': kg_context,
                            'metadata': {
                                'score': 0.85,
                                'type': 'kg_structured',
                                'mode': kg_result.get("mode", "local")
                            }
                        })
                    
                    n_src = len(kg_result.get("source_chunks", []))
                    n_ent = len(kg_result.get("entities", []))
                    print(f"[Retriever] KG structured: {n_src} source chunks, "
                          f"{n_ent} entities, {len(kg_entity_names)} names for expansion")
                          
                except Exception as e:
                    print(f"[Retriever] KG retrieval failed: {e}")
            elif mode in ["kg", "kg-contextual"]:
                print("[Retriever] Warning: KG-only mode but KG retriever not initialized")
        
        # --- Step 2: Hybrid retrieval (with optional query expansion) ---
        hybrid_nodes = []
        if mode in ["hybrid", "hybrid-kg", "hybrid-kg-contextual"]:
            # Query expansion: append top KG entity names to improve BM25/vector recall
            expanded_query = query
            if kg_entity_names:
                # Use top 5 most relevant entity names
                expansion_terms = " ".join(kg_entity_names[:5])
                expanded_query = f"{query} {expansion_terms}"
                print(f"[Retriever] Query expanded with KG entities: +[{expansion_terms}]")
            
            # Initial retrieval (BM25 + Vector)
            hybrid_nodes = self.retriever.retrieve(expanded_query)
            
        # --- Step 3: Combine, Deduplicate, and Rerank ---
        
        # Convert KG results to NodeWithScore format
        kg_nodes = []
        for res in results:
             # Create a TextNode
             node = TextNode(text=res['page_content'], metadata=res['metadata'])
             # Create NodeWithScore (using the initial score assigned in Step 1)
             kg_nodes.append(NodeWithScore(node=node, score=res['metadata']['score']))
             
        # Combine all candidates
        all_candidates = hybrid_nodes + kg_nodes
        
        # Deduplicate by content
        seen_content = set()
        unique_candidates = []
        for node in all_candidates:
            content = node.node.get_content().strip()
            if content and content not in seen_content:
                seen_content.add(content)
                unique_candidates.append(node)
        
        print(f"[Retriever] Merging results: {len(hybrid_nodes)} Hybrid + {len(kg_nodes)} KG -> {len(unique_candidates)} Unique")
        
        # Rerank everything together
        # We use the ORIGINAL query for reranking to ensure relevance
        final_nodes = self.reranker_module.rerank(unique_candidates, query)
        
        # Return top_k
        final_nodes = final_nodes[:top_k]
        
        # Convert back to dict format for main.py
        final_results = [
            {
                'page_content': n.node.get_content(),
                'metadata': {
                    'score': n.score,
                    'type': n.node.metadata.get('type', 'hybrid_retrieval'), # Keep original type if exists
                    **n.node.metadata
                }
            } for n in final_nodes
        ]
        
        return final_results


def create_retriever(chunks, language, chunksize, similarity_threshold=0.5, use_kg=False, contextual_kg=False):
    return Retriever(chunks, language, chunksize, similarity_threshold, use_kg=use_kg, contextual_kg=contextual_kg)

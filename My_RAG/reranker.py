"""
Cross-encoder Reranker using sentence-transformers.
Uses cross-encoder for relevance scoring between query and passages.
"""
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
import os


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3, use_fp16=False):
        """
        初始化 Cross-Encoder Reranker
        :param model_name: 使用的模型名稱 (預設 cross-encoder/ms-marco-MiniLM-L-6-v2)
        :param top_n: 重排序後要保留前幾名
        :param use_fp16: 是否開啟半精度 (節省記憶體)
        """
        self.top_n = top_n
        # cross-encoder/ms-marco-MiniLM-L-6-v2 is lightweight and effective
        self.model = CrossEncoder(model_name, max_length=512)
        print(f"[Reranker] Loaded CrossEncoder: {model_name}")

    def rerank(self, nodes, query):
        """
        執行重排序
        :param nodes: 初步檢索到的節點列表 (LlamaIndex NodeWithScore)
        :param query: 使用者的查詢字串
        :return: 排序後的節點列表
        """
        if not nodes:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, node.node.get_content()) for node in nodes]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Combine nodes with new scores and sort
        scored_nodes = []
        for i, node in enumerate(nodes):
            new_node = NodeWithScore(node=node.node, score=float(scores[i]))
            scored_nodes.append(new_node)
        
        # Sort by score descending and return top_n
        scored_nodes.sort(key=lambda x: x.score, reverse=True)
        return scored_nodes[:self.top_n]

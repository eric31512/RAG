
import sys
import os

# Ensure we can import from My_RAG
sys.path.append(os.path.join(os.getcwd(), 'My_RAG'))

from My_RAG.retriever import create_retriever
from llama_index.core.schema import TextNode

def test_retriever():
    print("Initializing Retriever with KG enabled...")
    # Assuming 'en' and chunksize 512 are available/default
    # We use 'hybrid-kg' mode to test the merge logic
    retriever = create_retriever(
        chunks=[], # We might not need chunks if we just want to test the retrieval logic with existing indices
        language="en", 
        chunksize=512, 
        use_kg=True, 
        contextual_kg=False
    )
    
    query = "What is the main topic of the document?"
    print(f"\nQuerying: '{query}'")
    
    try:
        results = retriever.retrieve(query, top_k=5, mode="hybrid-kg")
        
        print(f"\nRetrieved {len(results)} results:")
        for i, res in enumerate(results):
            meta = res['metadata']
            print(f"[{i+1}] Score: {meta.get('score', 'N/A'):.4f} | Type: {meta.get('type', 'N/A')} | Content: {res['page_content'][:50]}...")
            
    except Exception as e:
        print(f"\nError during retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retriever()

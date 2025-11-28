from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    # Initialize embedding model for semantic chunking
    # Using the same model as the retriever for consistency
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2") #Needs to be replaced with the embedding model provided by the TA
    
    # Initialize Semantic Chunker
    # breakpoint_threshold_type="percentile" is a good starting point
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    chunks = []
    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            text = doc['content']
            lang = doc['language']
            
            # Only apply semantic chunking to the target language (or all if desired)
            # Here we apply it to the specified language, falling back to original logic or just skipping others
            if lang == language:
                # Create semantic chunks
                semantic_docs = semantic_chunker.create_documents([text])
                
                for i, semantic_doc in enumerate(semantic_docs):
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop('content', None)
                    chunk_metadata['chunk_index'] = i
                    
                    chunk = {
                        'page_content': semantic_doc.page_content,
                        'metadata': chunk_metadata
                    }
                    chunks.append(chunk)
                    
    return chunks
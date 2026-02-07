"""
Retriever module that combines embedding and vector search.
"""

import sys
sys.path.append('src')

from retrieval.embedder import TextEmbedder
from retrieval.vector_store import VectorStore
from data_processing.load_spider import SpiderDataLoader


class Retriever:
    """
    High-level retrieval interface.
    
    Retrieves relevant examples and schemas for a user question.
    """
    
    def __init__(self, embedder: TextEmbedder, vector_store: VectorStore, 
                 data_loader: SpiderDataLoader):
        """
        Initialize retriever.
        
        Args:
            embedder: TextEmbedder for creating query embeddings
            vector_store: VectorStore with built index
            data_loader: SpiderDataLoader for schema formatting
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.data_loader = data_loader
    
    def retrieve(self, question: str, k: int = 5, db_id: str = None):
        """
        Retrieve top-k most relevant items for a question.
        
        Args:
            question: User's natural language question
            k: Number of items to retrieve
            db_id: Optional database ID to filter results
            
        Returns:
            List of retrieved items with metadata
        """
        # Create embedding for the question
        query_embedding = self.embedder.embed_text(question)
        
        # If db_id specified, retrieve more and filter
        retrieve_k = k * 5 if db_id else k
        
        # Search vector store
        distances, metadata_list = self.vector_store.search(query_embedding, k=retrieve_k)
        
        # Add similarity scores
        results = []
        for distance, metadata in zip(distances, metadata_list):
            result = metadata.copy()
            result['distance'] = distance
            result['similarity'] = 1.0 / (1.0 + distance)
            results.append(result)
        
        # Filter by database if specified
        if db_id:
            results = [r for r in results if r['db_id'] == db_id]
            results = results[:k]
        
        return results
    
    def format_context(self, question: str, db_id: str, k: int = 5, 
                      include_schema: bool = True):
        """
        Format retrieved context for LLM prompt.
        
        Args:
            question: User's question
            db_id: Database ID (to get correct schema and examples)
            k: Number of examples to retrieve
            include_schema: Whether to include database schema
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add schema for the specific database
        if include_schema:
            schema = self.data_loader.format_schema_text(db_id)
            context_parts.append("DATABASE SCHEMA:")
            context_parts.append(schema)
            context_parts.append("")
        
        # Retrieve examples from the same database
        results = self.retrieve(question, k=k, db_id=db_id)
        
        # Add example queries
        if results:
            context_parts.append("EXAMPLE QUERIES FROM THIS DATABASE:")
            context_parts.append("")
            
            for i, result in enumerate(results, 1):
                context_parts.append(f"Example {i}:")
                context_parts.append(f"Question: {result['question']}")
                context_parts.append(f"SQL: {result['query']}")
                context_parts.append("")
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    # Test the retriever
    from pathlib import Path
    
    print("Loading components...")
    
    # Load data
    loader = SpiderDataLoader(r"F:\text2sql\spider_data")
    
    # Load embedder
    embedder = TextEmbedder()
    
    # Load vector store
    vector_store = VectorStore(embedder.get_embedding_dim())
    vector_store.load(r"F:\text2sql\indices")
    
    # Create retriever
    retriever = Retriever(embedder, vector_store, loader)
    
    # Test retrieval with specific database
    test_question = "How many singers do we have?"
    test_db = "concert_singer"
    
    print(f"\nTest question: {test_question}")
    print(f"Database: {test_db}")
    print("=" * 60)
    
    # Get formatted context for specific database
    context = retriever.format_context(test_question, test_db, k=3, include_schema=True)
    print(context)

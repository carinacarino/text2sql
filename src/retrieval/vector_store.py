"""
Vector store module using FAISS for efficient similarity search.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    """
    Manages FAISS index for vector similarity search.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
    
    def build_index(self, embeddings, metadata):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Matrix of embeddings with shape (num_items, embedding_dim)
            metadata: List of metadata dicts, one for each embedding
        """
        print(f"Building FAISS index with {embeddings.shape[0]} vectors...")
        
        # Use L2 distance
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata = metadata
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query_embedding, k: int = 5):
        """
        Search for k most similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (distances, metadata)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get metadata for retrieved items
        retrieved_metadata = [self.metadata[idx] for idx in indices[0]]
        
        return distances[0].tolist(), retrieved_metadata
    
    def save(self, save_path: str):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            save_path: Directory path to save index and metadata
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / 'faiss_index.bin'
        faiss.write_index(self.index, str(index_file))
        print(f"Index saved to {index_file}")
        
        # Save metadata
        metadata_file = save_path / 'metadata.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Metadata saved to {metadata_file}")
    
    def load(self, load_path: str):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            load_path: Directory path containing saved index
        """
        load_path = Path(load_path)
        
        # Load FAISS index
        index_file = load_path / 'faiss_index.bin'
        self.index = faiss.read_index(str(index_file))
        print(f"Index loaded from {index_file}")
        
        # Load metadata
        metadata_file = load_path / 'metadata.pkl'
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"Metadata loaded, {len(self.metadata)} items")


if __name__ == "__main__":
    # Test with embedder
    import sys
    sys.path.append('src')
    from retrieval.embedder import TextEmbedder
    
    # Create embeddings
    embedder = TextEmbedder()
    texts = [
        "How many singers do we have?",
        "What are the names of all singers?",
        "Show me the concert dates",
        "List all stadium names"
    ]
    embeddings = embedder.embed_batch(texts, show_progress=False)
    
    # Create metadata
    metadata = [{'text': text, 'id': i} for i, text in enumerate(texts)]
    
    # Build index
    store = VectorStore(embedder.get_embedding_dim())
    store.build_index(embeddings, metadata)
    
    # Search
    query = "How many concerts are there?"
    query_embedding = embedder.embed_text(query)
    distances, results = store.search(query_embedding, k=2)
    
    print(f"\nQuery: {query}")
    print("Top 2 results:")
    for i, (dist, meta) in enumerate(zip(distances, results)):
        print(f"{i+1}. {meta['text']} (distance: {dist:.4f})")

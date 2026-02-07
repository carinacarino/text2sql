"""
Text embedding module using sentence-transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Creates embeddings for text using sentence-transformers.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       'all-MiniLM-L6-v2' is fast and good for prototyping (384 dim)
                       'all-mpnet-base-v2' is better quality but slower (768 dim)
        """
        print(f"Loading sentence-transformer model: {model_name}")
        
        # Check for GPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded on {device.upper()}. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str):
        """
        Create embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: list, batch_size: int = 32, show_progress: bool = True):
        """
        Create embeddings for multiple texts efficiently.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Matrix of embeddings with shape (num_texts, embedding_dim)
        """
        if show_progress:
            print(f"Creating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def get_embedding_dim(self):
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim


if __name__ == "__main__":
    # Test the embedder
    embedder = TextEmbedder()
    
    # Test single text
    text = "How many singers do we have?"
    embedding = embedder.embed_text(text)
    print(f"\nSingle text embedding shape: {embedding.shape}")
    
    # Test batch
    texts = [
        "How many singers do we have?",
        "What are the names of all singers?",
        "Show me the concert dates"
    ]
    embeddings = embedder.embed_batch(texts, show_progress=False)
    print(f"Batch embeddings shape: {embeddings.shape}")

"""
Script to build FAISS index from Spider dataset.

This script:
1. Loads Spider dataset
2. Creates retrieval corpus (training examples)
3. Generates embeddings
4. Builds FAISS index
5. Saves index to disk

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_processing.load_spider import SpiderDataLoader
from retrieval.embedder import TextEmbedder
from retrieval.vector_store import VectorStore


def build_index():
    """Build FAISS index from Spider dataset."""
    
    print("=" * 60)
    print("Building FAISS Index for Text-to-SQL RAG")
    print("=" * 60)
    
    # Configuration
    SPIDER_PATH = r"F:\text2sql\spider_data"
    INDEX_PATH = r"F:\text2sql\indices"
    MODEL_NAME = "all-MiniLM-L6-v2"
    BATCH_SIZE = 32
    
    # Step 1: Load Spider dataset
    print("\n[1/4] Loading Spider dataset...")
    loader = SpiderDataLoader(SPIDER_PATH)
    train_examples = loader.get_train_examples()
    print(f"Loaded {len(train_examples)} training examples")
    
    # Step 2: Prepare corpus
    print("\n[2/4] Preparing corpus for retrieval...")
    corpus = []
    
    # Add each training example
    for example in train_examples:
        item = {
            'type': 'example',
            'db_id': example['db_id'],
            'question': example['question'],
            'query': example['query'],
            'text_to_embed': example['question']  # Embed the question
        }
        corpus.append(item)
    
    print(f"Created corpus with {len(corpus)} items")
    
    # Step 3: Generate embeddings
    print(f"\n[3/4] Generating embeddings (model: {MODEL_NAME})...")
    embedder = TextEmbedder(model_name=MODEL_NAME)
    
    texts_to_embed = [item['text_to_embed'] for item in corpus]
    embeddings = embedder.embed_batch(texts_to_embed, batch_size=BATCH_SIZE)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Prepare metadata (remove text_to_embed to save space)
    metadata = []
    for item in corpus:
        meta = item.copy()
        meta.pop('text_to_embed')
        metadata.append(meta)
    
    # Step 4: Build and save FAISS index
    print("\n[4/4] Building FAISS index...")
    vector_store = VectorStore(embedder.get_embedding_dim())
    vector_store.build_index(embeddings, metadata)
    
    print(f"\nSaving index to {INDEX_PATH}...")
    vector_store.save(INDEX_PATH)
    
    print("\n" + "=" * 60)
    print("Index building complete!")
    print("=" * 60)
    print(f"Index location: {INDEX_PATH}")
    print(f"Total vectors: {vector_store.index.ntotal}")
    print(f"Embedding dimension: {embedder.get_embedding_dim()}")
    
    # Test the index
    print("\n" + "=" * 60)
    print("Testing retrieval...")
    print("=" * 60)
    
    test_question = "How many singers do we have?"
    print(f"\nTest question: {test_question}")
    
    query_embedding = embedder.embed_text(test_question)
    distances, results = vector_store.search(query_embedding, k=3)
    
    print("\nTop 3 results:")
    for i, (dist, result) in enumerate(zip(distances, results), 1):
        print(f"\n{i}. Database: {result['db_id']}")
        print(f"   Question: {result['question']}")
        print(f"   SQL: {result['query']}")
        print(f"   Distance: {dist:.4f}")


if __name__ == "__main__":
    build_index()

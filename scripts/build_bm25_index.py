"""
Script to build BM25 index from Spider training data.
"""

import sys
sys.path.append('src')

from data_processing.load_spider import SpiderDataLoader
from retrieval.bm25_retriever import BM25Retriever
import pickle
from pathlib import Path

print("=" * 60)
print("Building BM25 Index")
print("=" * 60)

# Load Spider data
print("\n[1/3] Loading Spider dataset...")
loader = SpiderDataLoader(r"F:\text2sql\spider_data")
train_examples = loader.get_train_examples()
print(f"Loaded {len(train_examples)} training examples")

# Create corpus
print("\n[2/3] Creating corpus...")
corpus = []
for example in train_examples:
    item = {
        'type': 'example',
        'db_id': example['db_id'],
        'question': example['question'],
        'query': example['query'],
        'text_to_embed': example['question']
    }
    corpus.append(item)
print(f"Created corpus with {len(corpus)} items")

# Build BM25 index
print("\n[3/3] Building BM25 index...")
bm25_retriever = BM25Retriever(corpus)

# Save to disk
output_path = Path(r"F:\text2sql\indices\bm25_index.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(bm25_retriever, f)

print(f"\nBM25 index saved to {output_path}")
print("=" * 60)
print("Done!")

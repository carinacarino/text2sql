"""
BM25 retriever for keyword-based retrieval.
"""

from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:
    """
    BM25-based retriever for keyword matching.
    
    BM25 is a classic information retrieval algorithm that ranks
    documents based on term frequency and inverse document frequency.
    """
    
    def __init__(self, corpus):
        """
        Initialize BM25 retriever.
        
        Args:
            corpus: List of dictionaries with 'text_to_embed' and metadata
        """
        # Tokenize questions
        self.corpus = corpus
        tokenized_corpus = [item['text_to_embed'].lower().split() for item in corpus]
        
        # Build BM25 index
        print(f"Building BM25 index with {len(corpus)} items...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built!")
    
    def retrieve(self, question: str, k: int = 5, db_id: str = None):
        """
        Retrieve top-k most relevant items using BM25.
        
        Args:
            question: Query question
            k: Number of items to retrieve
            db_id: Optional database ID to filter results
            
        Returns:
            List of retrieved items with metadata and scores
        """
        # Tokenize query
        tokenized_query = question.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        if db_id:
            # Filter by database first
            filtered_indices = [i for i, item in enumerate(self.corpus) 
                              if item['db_id'] == db_id]
            filtered_scores = [(i, scores[i]) for i in filtered_indices]
            top_indices = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:k]
            top_indices = [i for i, _ in top_indices]
        else:
            top_indices = np.argsort(scores)[::-1][:k]
        
        # Get results
        results = []
        for idx in top_indices:
            result = self.corpus[idx].copy()
            result['score'] = float(scores[idx])
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test BM25 retriever
    import sys
    sys.path.append('src')
    from data_processing.load_spider import SpiderDataLoader
    
    # Load data
    loader = SpiderDataLoader(r"F:\text2sql\spider_data")
    
    # Create corpus
    corpus = []
    for example in loader.get_train_examples()[:100]:  # Test with 100
        item = {
            'type': 'example',
            'db_id': example['db_id'],
            'question': example['question'],
            'query': example['query'],
            'text_to_embed': example['question']
        }
        corpus.append(item)
    
    # Build BM25 retriever
    retriever = BM25Retriever(corpus)
    
    # Test query
    question = "How many singers do we have?"
    db_id = "concert_singer"
    
    print(f"\nQuery: {question}")
    print(f"Database: {db_id}")
    print("=" * 60)
    
    results = retriever.retrieve(question, k=3, db_id=db_id)
    
    print("\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Question: {result['question']}")
        print(f"   SQL: {result['query']}")

"""
Complete Text-to-SQL pipeline combining retrieval and generation.
"""

import sys
sys.path.append('src')

from data_processing.load_spider import SpiderDataLoader
from retrieval.embedder import TextEmbedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from generation.ollama_client import OllamaClient
from generation.prompt_builder import PromptBuilder
from generation.cot_prompt_builder import CoTPromptBuilder


class Text2SQLPipeline:
    """
    Complete pipeline for text-to-SQL with optional RAG.
    """
    
    def __init__(self, spider_path: str, index_path: str, 
                 model_name: str = "codellama"):
        """
        Initialize the pipeline.
        
        Args:
            spider_path: Path to Spider dataset
            index_path: Path to FAISS index
            model_name: Ollama model name
        """
        print("Initializing Text-to-SQL pipeline...")
        
        # Load components
        self.data_loader = SpiderDataLoader(spider_path)
        self.embedder = TextEmbedder()
        
        self.vector_store = VectorStore(self.embedder.get_embedding_dim())
        self.vector_store.load(index_path)
        
        self.retriever = Retriever(self.embedder, self.vector_store, self.data_loader)
        self.ollama_client = OllamaClient(model_name=model_name)
        
        print("Pipeline initialized!")
    
    def generate_sql_baseline(self, question: str):
        """
        Generate SQL without RAG (baseline).
        
        Args:
            question: Natural language question
            
        Returns:
            Generated SQL query
        """
        prompt = PromptBuilder.build_baseline_prompt(question)
        sql = self.ollama_client.generate_sql(prompt)
        return sql
    
    def generate_sql_rag(self, question: str, db_id: str, k: int = 5, 
                        include_schema: bool = True, include_examples: bool = True,
                        use_cot: bool = False):
        """
        Generate SQL with RAG (retrieval-augmented generation).
        
        Args:
            question: Natural language question
            db_id: Database ID (to retrieve correct schema and examples)
            k: Number of examples to retrieve
            include_schema: Whether to include database schema
            include_examples: Whether to include example queries
            use_cot: Whether to use Chain-of-Thought prompting
            
        Returns:
            Generated SQL query
        """
        # Build context based on configuration
        context_parts = []
        
        # Add schema if requested
        if include_schema:
            schema = self.data_loader.format_schema_text(db_id)
            context_parts.append("DATABASE SCHEMA:")
            context_parts.append(schema)
            context_parts.append("")
        
        # Add examples if requested
        if include_examples:
            results = self.retriever.retrieve(question, k=k, db_id=db_id)
            if results:
                context_parts.append("EXAMPLE QUERIES FROM THIS DATABASE:")
                context_parts.append("")
                for i, result in enumerate(results, 1):
                    context_parts.append(f"Example {i}:")
                    context_parts.append(f"Question: {result['question']}")
                    context_parts.append(f"SQL: {result['query']}")
                    context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Build prompt (with or without CoT)
        if use_cot:
            prompt = CoTPromptBuilder.build_cot_prompt(question, context)
        else:
            prompt = PromptBuilder.build_rag_prompt(question, context)
        
        # Generate SQL
        sql = self.ollama_client.generate_sql(prompt)
        return sql


if __name__ == "__main__":
    # Test the pipeline
    print("=" * 60)
    print("Testing Text-to-SQL Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = Text2SQLPipeline(
        spider_path=r"F:\text2sql\spider_data",
        index_path=r"F:\text2sql\indices"
    )
    
    # Test question
    question = "How many singers do we have?"
    db_id = "concert_singer"
    print(f"\nQuestion: {question}")
    print(f"Database: {db_id}\n")
    
    # Test different configurations
    print("1. BASELINE (No RAG):")
    print("-" * 60)
    baseline_sql = pipeline.generate_sql_baseline(question)
    print(baseline_sql)
    print()
    
    print("2. SCHEMA ONLY:")
    print("-" * 60)
    schema_sql = pipeline.generate_sql_rag(question, db_id, include_schema=True, include_examples=False)
    print(schema_sql)
    print()
    
    print("3. EXAMPLES ONLY:")
    print("-" * 60)
    examples_sql = pipeline.generate_sql_rag(question, db_id, include_schema=False, include_examples=True, k=3)
    print(examples_sql)
    print()
    
    print("4. FULL RAG (Schema + Examples):")
    print("-" * 60)
    full_rag_sql = pipeline.generate_sql_rag(question, db_id, include_schema=True, include_examples=True, k=3)
    print(full_rag_sql)

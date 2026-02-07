"""
Prompt builder for SQL generation.
"""


class PromptBuilder:
    """
    Builds prompts for SQL generation with optional RAG context.
    """
    
    @staticmethod
    def build_baseline_prompt(question: str):
        """
        Build baseline prompt without retrieval (no RAG).
        
        Args:
            question: User's natural language question
            
        Returns:
            Prompt string
        """
        prompt = f"""Generate a SQL query for the following question.
Return ONLY the SQL query, nothing else.

Question: {question}

SQL:"""
        return prompt
    
    @staticmethod
    def build_rag_prompt(question: str, context: str):
        """
        Build RAG prompt with retrieved context.
        
        Args:
            question: User's natural language question
            context: Retrieved context (schema + examples)
            
        Returns:
            Prompt string
        """
        prompt = f"""You are a SQL expert. Generate a SQL query for the question below.

{context}

Now generate SQL for this question. Return ONLY the SQL query, nothing else.

Question: {question}

SQL:"""
        return prompt


if __name__ == "__main__":
    # Test prompt building
    
    # Baseline prompt
    question = "How many singers do we have?"
    baseline = PromptBuilder.build_baseline_prompt(question)
    
    print("BASELINE PROMPT:")
    print("=" * 60)
    print(baseline)
    print()
    
    # RAG prompt
    context = """DATABASE SCHEMA:
Database: concert_singer

Tables and Columns:
singer:
  - Singer_ID (number)
  - Name (text)
  - Age (number)

EXAMPLE QUERIES:

Example 1:
Question: How many artists do we have?
SQL: SELECT count(*) FROM artist"""
    
    rag = PromptBuilder.build_rag_prompt(question, context)
    
    print("RAG PROMPT:")
    print("=" * 60)
    print(rag)

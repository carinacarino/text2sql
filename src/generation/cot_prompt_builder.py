"""
Chain-of-Thought prompt builder for SQL generation.
"""


class CoTPromptBuilder:
    """
    Builds prompts with Chain-of-Thought reasoning for better SQL generation.
    """
    
    @staticmethod
    def build_cot_prompt(question: str, context: str):
        """
        Build Chain-of-Thought prompt with retrieved context.
        
        Args:
            question: User's natural language question
            context: Retrieved context (schema + examples)
            
        Returns:
            Prompt string with CoT instructions
        """
        prompt = f"""You are a SQL expert. Generate a SQL query for the question below.

{context}

Question: {question}

Let's think step by step:
1. Which tables do I need?
2. What columns should I select?
3. Do I need any JOINs? (Only if querying multiple tables)
4. What conditions go in the WHERE clause?
5. Do I need GROUP BY, ORDER BY, or LIMIT?

After thinking through these steps, provide ONLY the final SQL query with no explanation.

SQL:"""
        return prompt


if __name__ == "__main__":
    # Test CoT prompt
    question = "How many singers from France?"
    context = """DATABASE SCHEMA:
Database: concert_singer

Tables and Columns:
singer:
  - Singer_ID (number)
  - Name (text)
  - Country (text)
  - Age (number)

EXAMPLE QUERIES:

Example 1:
Question: How many artists do we have?
SQL: SELECT count(*) FROM artist"""
    
    cot_prompt = CoTPromptBuilder.build_cot_prompt(question, context)
    
    print("CHAIN-OF-THOUGHT PROMPT:")
    print("=" * 60)
    print(cot_prompt)

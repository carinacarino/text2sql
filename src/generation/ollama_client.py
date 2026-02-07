"""
Ollama client for SQL generation.
"""

import requests
import re


class OllamaClient:
    """
    Client for interacting with Ollama API to generate SQL queries.
    """
    
    def __init__(self, model_name: str = "codellama", 
                 base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def extract_sql(self, text: str):
        """
        Extract SQL query from generated text.
        
        Handles cases where the model adds explanation or formatting.
        
        Args:
            text: Generated text that may contain SQL
            
        Returns:
            Extracted SQL query
        """
        # Remove markdown code blocks
        text = re.sub(r'```sql\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Look for SELECT/WITH/INSERT/UPDATE/DELETE until we hit explanatory text
        # Stop at patterns like "This query", "The above", etc.
        sql_pattern = r'((?:WITH|SELECT|INSERT|UPDATE|DELETE)\s+.*?)(?=\n(?:This |The |Note:|Explanation:|--)|$)'
        match = re.search(sql_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            sql = match.group(1).strip()
        else:
            # Fallback: take everything until first explanation sentence
            lines = text.strip().split('\n')
            sql_lines = []
            for line in lines:
                line = line.strip()
                # Stop at explanation sentences
                if line and any(line.startswith(phrase) for phrase in 
                              ['This query', 'The query', 'This ', 'Note:', 'Explanation:']):
                    break
                if line:
                    sql_lines.append(line)
            sql = '\n'.join(sql_lines) if sql_lines else text.strip()
        
        # Remove trailing semicolons and whitespace
        sql = sql.rstrip(';').strip()
        
        return sql
    
    def generate_sql(self, prompt: str, temperature: float = 0.0):
        """
        Generate SQL query from prompt.
        
        Args:
            prompt: Full prompt with context and question
            temperature: Sampling temperature (0.0 = deterministic)
            
        Returns:
            Generated SQL query as string
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            # Extract just the SQL
            sql = self.extract_sql(generated_text)
            
            return sql
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return None
    
    def is_available(self):
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False


if __name__ == "__main__":
    # Test SQL extraction
    client = OllamaClient()
    
    test_cases = [
        "SELECT count(*) FROM users",
        "```sql\nSELECT * FROM table\n```",
        "Here's the query:\nSELECT id FROM data\nThis will return all IDs.",
        "SELECT COUNT(*) FROM singers;"
    ]
    
    print("Testing SQL extraction:")
    for text in test_cases:
        extracted = client.extract_sql(text)
        print(f"\nInput: {repr(text)}")
        print(f"Extracted: {repr(extracted)}")

"""
Module for loading and processing Spider dataset.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class SpiderDataLoader:
    """
    Loads and processes the Spider text-to-SQL dataset.
    
    The Spider dataset contains:
    - train_spider.json: Training examples with questions and SQL queries
    - dev.json: Development/evaluation set
    - tables.json: Database schema information
    - database/: SQLite database files
    """
    
    def __init__(self, spider_data_path: str):
        """
        Initialize the Spider data loader.
        
        Args:
            spider_data_path: Path to the Spider dataset directory
        """
        self.data_path = Path(spider_data_path)
        
        # Verify required files exist
        required_files = ['train_spider.json', 'dev.json', 'tables.json']
        for file in required_files:
            if not (self.data_path / file).exists():
                raise FileNotFoundError(f"Required file not found: {file}")
        
        # Load data
        self.train_data = self._load_json('train_spider.json')
        self.dev_data = self._load_json('dev.json')
        self.tables_data = self._load_json('tables.json')
        
        # Create lookup dictionaries for fast access
        self.db_schemas = {db['db_id']: db for db in self.tables_data}
    
    def _load_json(self, filename: str) -> List[Dict]:
        """
        Load JSON file from the Spider dataset.
        
        Args:
            filename: Name of the JSON file to load
            
        Returns:
            Loaded JSON data as list or dict
        """
        with open(self.data_path / filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_train_examples(self) -> List[Dict]:
        """
        Get all training examples.
        
        Returns:
            List of training examples with questions and SQL queries
        """
        return self.train_data
    
    def get_dev_examples(self) -> List[Dict]:
        """
        Get all development/evaluation examples.
        
        Returns:
            List of dev examples with questions and SQL queries
        """
        return self.dev_data
    
    def get_schema(self, db_id: str) -> Dict:
        """
        Get schema information for a specific database.
        
        Args:
            db_id: Database identifier
            
        Returns:
            Schema information including tables, columns, types, and foreign keys
        """
        if db_id not in self.db_schemas:
            raise ValueError(f"Database {db_id} not found in schema data")
        return self.db_schemas[db_id]
    
    def format_schema_text(self, db_id: str) -> str:
        """
        Format database schema as human-readable text for LLM context.
        
        This creates a text representation of the schema that can be included
        in prompts to help the LLM understand the database structure.
        
        Args:
            db_id: Database identifier
            
        Returns:
            Formatted schema text
        """
        schema = self.get_schema(db_id)
        
        # Start with database name
        schema_text = f"Database: {db_id}\n\n"
        
        # Add tables and columns
        schema_text += "Tables and Columns:\n"
        for table_idx, table_name in enumerate(schema['table_names_original']):
            schema_text += f"\n{table_name}:\n"
            
            # Get columns for this table
            for col_idx, (col_table_idx, col_name) in enumerate(schema['column_names_original']):
                if col_table_idx == table_idx:
                    col_type = schema['column_types'][col_idx]
                    schema_text += f"  - {col_name} ({col_type})\n"
        
        # Add foreign keys if they exist
        if 'foreign_keys' in schema and schema['foreign_keys']:
            schema_text += "\nForeign Keys:\n"
            for from_col_idx, to_col_idx in schema['foreign_keys']:
                from_table_idx, from_col = schema['column_names_original'][from_col_idx]
                to_table_idx, to_col = schema['column_names_original'][to_col_idx]
                from_table = schema['table_names_original'][from_table_idx]
                to_table = schema['table_names_original'][to_table_idx]
                schema_text += f"  - {from_table}.{from_col} -> {to_table}.{to_col}\n"
        
        # Add primary keys if they exist
        if 'primary_keys' in schema and schema['primary_keys']:
            schema_text += "\nPrimary Keys:\n"
            for pk_idx in schema['primary_keys']:
                table_idx, col_name = schema['column_names_original'][pk_idx]
                table_name = schema['table_names_original'][table_idx]
                schema_text += f"  - {table_name}.{col_name}\n"
        
        return schema_text
    
    def get_examples_by_database(self, db_id: str, split: str = 'train') -> List[Dict]:
        """
        Get all examples for a specific database.
        
        Args:
            db_id: Database identifier
            split: 'train' or 'dev'
            
        Returns:
            List of examples for the specified database
        """
        data = self.train_data if split == 'train' else self.dev_data
        return [ex for ex in data if ex['db_id'] == db_id]
    
    def get_database_path(self, db_id: str) -> Path:
        """
        Get the path to a SQLite database file.
        
        Args:
            db_id: Database identifier
            
        Returns:
            Path to the SQLite database file
        """
        db_path = self.data_path / 'database' / db_id / f'{db_id}.sqlite'
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        return db_path
    
    def get_all_databases(self) -> List[str]:
        """
        Get list of all database IDs in the dataset.
        
        Returns:
            List of database identifiers
        """
        return list(self.db_schemas.keys())
    
    def create_retrieval_corpus(self) -> List[Dict]:
        """
        Create corpus of items for retrieval system.
        
        This combines training examples with their schemas to create
        a searchable corpus for the RAG system.
        
        Returns:
            List of retrieval items with text, metadata, and context
        """
        corpus = []
        
        # Add training examples with their schemas
        for example in self.train_data:
            db_id = example['db_id']
            schema_text = self.format_schema_text(db_id)
            
            item = {
                'type': 'example',
                'db_id': db_id,
                'question': example['question'],
                'query': example['query'],
                'schema': schema_text,
                # Text to embed - question + schema context
                'text_to_embed': f"{example['question']}\n\nDatabase: {db_id}"
            }
            corpus.append(item)
        
        # Add schema-only items for each database
        for db_id in self.get_all_databases():
            schema_text = self.format_schema_text(db_id)
            
            item = {
                'type': 'schema',
                'db_id': db_id,
                'schema': schema_text,
                # Text to embed - database name and schema
                'text_to_embed': f"Database: {db_id}\n{schema_text}"
            }
            corpus.append(item)
        
        return corpus
    
    def __repr__(self) -> str:
        """String representation of the data loader."""
        return (f"SpiderDataLoader(train_examples={len(self.train_data)}, "
                f"dev_examples={len(self.dev_data)}, "
                f"databases={len(self.db_schemas)})")


if __name__ == "__main__":
    # Example usage
    loader = SpiderDataLoader(r"F:\text2sql\spider_data")
    print(loader)
    
    # Print first example
    print("\nFirst training example:")
    example = loader.get_train_examples()[0]
    print(f"Question: {example['question']}")
    print(f"SQL: {example['query']}")
    print(f"Database: {example['db_id']}")
    
    # Print formatted schema
    print("\nFormatted schema:")
    print(loader.format_schema_text(example['db_id']))

"""
SQL executor for running queries on Spider databases.
"""

import sqlite3
from pathlib import Path


class SQLExecutor:
    """
    Executes SQL queries on Spider SQLite databases.
    """
    
    def __init__(self, spider_data_path: str):
        """
        Initialize SQL executor.
        
        Args:
            spider_data_path: Path to Spider dataset directory
        """
        self.spider_data_path = Path(spider_data_path)
        self.database_dir = self.spider_data_path / 'database'
    
    def get_database_path(self, db_id: str):
        """
        Get path to SQLite database file.
        
        Args:
            db_id: Database identifier
            
        Returns:
            Path to database file
        """
        db_path = self.database_dir / db_id / f'{db_id}.sqlite'
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        return db_path
    
    def execute_sql(self, sql: str, db_id: str):
        """
        Execute SQL query on a database.
        
        Args:
            sql: SQL query to execute
            db_id: Database identifier
            
        Returns:
            Tuple of (success, result_or_error)
            - If success: (True, list of result rows)
            - If error: (False, error message)
        """
        try:
            db_path = self.get_database_path(db_id)
            
            # Connect to database
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(sql)
            results = cursor.fetchall()
            
            conn.close()
            
            return True, results
            
        except Exception as e:
            return False, str(e)
    
    def compare_results(self, result1, result2):
        """
        Compare two query results for equality.
        
        Args:
            result1: First result set
            result2: Second result set
            
        Returns:
            True if results match, False otherwise
        """
        # Convert to sets for comparison (order doesn't matter for most queries)
        try:
            # Handle case where results are lists of tuples
            set1 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) 
                      for row in result1)
            set2 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) 
                      for row in result2)
            return set1 == set2
        except:
            # Fallback to direct comparison
            return result1 == result2


if __name__ == "__main__":
    # Test SQL executor
    executor = SQLExecutor(r"F:\text2sql\spider_data")
    
    # Test query
    sql = "SELECT count(*) FROM head WHERE age > 56"
    db_id = "department_management"
    
    print(f"Testing SQL execution...")
    print(f"Database: {db_id}")
    print(f"SQL: {sql}")
    
    success, result = executor.execute_sql(sql, db_id)
    
    if success:
        print(f"✓ Query executed successfully")
        print(f"Result: {result}")
    else:
        print(f"✗ Query failed")
        print(f"Error: {result}")

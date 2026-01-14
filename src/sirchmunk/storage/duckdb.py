"""
DuckDB database manager for Sirchmunk
Provides a comprehensive interface for DuckDB operations including
connection management, table operations, data manipulation, and analytics
"""

import duckdb
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class DuckDBManager:
    """
    A comprehensive DuckDB database manager providing common operations
    for data storage, retrieval, and analytics in the Sirchmunk system.
    """
    
    def __init__(self, db_path: Optional[str] = None, read_only: bool = False):
        """
        Initialize DuckDB connection
        
        Args:
            db_path: Path to database file. If None, creates in-memory database
            read_only: Whether to open database in read-only mode
        """
        self.db_path = db_path
        self.read_only = read_only
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            if self.db_path:
                self.connection = duckdb.connect(self.db_path, read_only=self.read_only)
                logger.info(f"Connected to DuckDB at {self.db_path}")
            else:
                self.connection = duckdb.connect(":memory:")
                logger.info("Connected to in-memory DuckDB")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("DuckDB connection closed")
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        try:
            self.connection.begin()
            yield self.connection
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
    
    def execute(self, query: str, parameters: Optional[List] = None) -> duckdb.DuckDBPyResult:
        """
        Execute SQL query
        
        Args:
            query: SQL query string
            parameters: Optional query parameters
            
        Returns:
            Query result
        """
        try:
            if parameters:
                return self.connection.execute(query, parameters)
            return self.connection.execute(query)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise
    
    def fetch_all(self, query: str, parameters: Optional[List] = None) -> List[Tuple]:
        """
        Execute query and fetch all results
        
        Args:
            query: SQL query string
            parameters: Optional query parameters
            
        Returns:
            List of result tuples
        """
        result = self.execute(query, parameters)
        return result.fetchall()
    
    def fetch_one(self, query: str, parameters: Optional[List] = None) -> Optional[Tuple]:
        """
        Execute query and fetch one result
        
        Args:
            query: SQL query string
            parameters: Optional query parameters
            
        Returns:
            Single result tuple or None
        """
        result = self.execute(query, parameters)
        return result.fetchone()
    
    def fetch_df(self, query: str, parameters: Optional[List] = None) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame
        
        Args:
            query: SQL query string
            parameters: Optional query parameters
            
        Returns:
            Results as DataFrame
        """
        result = self.execute(query, parameters)
        return result.df()
    
    def create_table(self, table_name: str, schema: Dict[str, str], if_not_exists: bool = True):
        """
        Create table with specified schema
        
        Args:
            table_name: Name of the table
            schema: Dictionary mapping column names to types
            if_not_exists: Whether to use IF NOT EXISTS clause
        """
        columns = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])
        if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""
        
        query = f"CREATE TABLE {if_not_exists_clause} {table_name} ({columns})"
        self.execute(query)
        logger.info(f"Table {table_name} created successfully")
    
    def drop_table(self, table_name: str, if_exists: bool = True):
        """
        Drop table
        
        Args:
            table_name: Name of the table to drop
            if_exists: Whether to use IF EXISTS clause
        """
        if_exists_clause = "IF EXISTS" if if_exists else ""
        query = f"DROP TABLE {if_exists_clause} {table_name}"
        self.execute(query)
        logger.info(f"Table {table_name} dropped successfully")
    
    def insert_data(self, table_name: str, data: Union[Dict, List[Dict], pd.DataFrame]):
        """
        Insert data into table
        
        Args:
            table_name: Target table name
            data: Data to insert (dict, list of dicts, or DataFrame)
        """
        if isinstance(data, dict):
            data = [data]
        
        if isinstance(data, list):
            if not data:
                return
            
            columns = list(data[0].keys())
            placeholders = ", ".join(["?" for _ in columns])
            column_names = ", ".join(columns)
            
            query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
            
            for row in data:
                values = [row.get(col) for col in columns]
                self.execute(query, values)
        
        elif isinstance(data, pd.DataFrame):
            # Use DuckDB's efficient DataFrame insertion
            self.connection.register("temp_df", data)
            self.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
            self.connection.unregister("temp_df")
        
        logger.info(f"Data inserted into {table_name}")
    
    def update_data(self, table_name: str, set_clause: Dict[str, Any], 
                   where_clause: str, where_params: Optional[List] = None):
        """
        Update data in table
        
        Args:
            table_name: Target table name
            set_clause: Dictionary of column-value pairs to update
            where_clause: WHERE condition
            where_params: Parameters for WHERE clause
        """
        set_parts = [f"{col} = ?" for col in set_clause.keys()]
        set_string = ", ".join(set_parts)
        
        query = f"UPDATE {table_name} SET {set_string} WHERE {where_clause}"
        params = list(set_clause.values())
        if where_params:
            params.extend(where_params)
        
        self.execute(query, params)
        logger.info(f"Data updated in {table_name}")
    
    def delete_data(self, table_name: str, where_clause: str, where_params: Optional[List] = None):
        """
        Delete data from table
        
        Args:
            table_name: Target table name
            where_clause: WHERE condition
            where_params: Parameters for WHERE clause
        """
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        self.execute(query, where_params)
        logger.info(f"Data deleted from {table_name}")
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_name = ?
        """
        result = self.fetch_one(query, [table_name])
        return result[0] > 0 if result else False
    
    def get_table_info(self, table_name: str) -> List[Dict]:
        """
        Get table schema information
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        query = f"DESCRIBE {table_name}"
        result = self.fetch_all(query)
        
        columns = []
        for row in result:
            columns.append({
                "column_name": row[0],
                "column_type": row[1],
                "null": row[2],
                "key": row[3] if len(row) > 3 else None,
                "default": row[4] if len(row) > 4 else None,
                "extra": row[5] if len(row) > 5 else None
            })
        
        return columns
    
    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows in table
        """
        query = f"SELECT COUNT(*) FROM {table_name}"
        result = self.fetch_one(query)
        return result[0] if result else 0
    
    def list_tables(self) -> List[str]:
        """
        Get list of all tables in database
        
        Returns:
            List of table names
        """
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        result = self.fetch_all(query)
        return [row[0] for row in result]
    
    def export_to_csv(self, table_name: str, file_path: str, delimiter: str = ","):
        """
        Export table data to CSV file
        
        Args:
            table_name: Source table name
            file_path: Output CSV file path
            delimiter: CSV delimiter
        """
        query = f"COPY {table_name} TO '{file_path}' (DELIMITER '{delimiter}', HEADER)"
        self.execute(query)
        logger.info(f"Table {table_name} exported to {file_path}")
    
    def import_from_csv(self, table_name: str, file_path: str, 
                       delimiter: str = ",", header: bool = True, 
                       create_table: bool = True):
        """
        Import data from CSV file
        
        Args:
            table_name: Target table name
            file_path: CSV file path
            delimiter: CSV delimiter
            header: Whether CSV has header row
            create_table: Whether to create table automatically
        """
        if create_table:
            # Let DuckDB auto-detect schema and create table
            query = f"""
            CREATE TABLE {table_name} AS 
            SELECT * FROM read_csv_auto('{file_path}', delim='{delimiter}', header={header})
            """
        else:
            # Insert into existing table
            query = f"""
            INSERT INTO {table_name} 
            SELECT * FROM read_csv_auto('{file_path}', delim='{delimiter}', header={header})
            """
        
        self.execute(query)
        logger.info(f"Data imported from {file_path} to {table_name}")
    
    def export_to_parquet(self, table_name: str, file_path: str):
        """
        Export table data to Parquet file
        
        Args:
            table_name: Source table name
            file_path: Output Parquet file path
        """
        query = f"COPY {table_name} TO '{file_path}' (FORMAT PARQUET)"
        self.execute(query)
        logger.info(f"Table {table_name} exported to Parquet: {file_path}")
    
    def import_from_parquet(self, table_name: str, file_path: str, create_table: bool = True):
        """
        Import data from Parquet file
        
        Args:
            table_name: Target table name
            file_path: Parquet file path
            create_table: Whether to create table automatically
        """
        if create_table:
            query = f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')"
        else:
            query = f"INSERT INTO {table_name} SELECT * FROM read_parquet('{file_path}')"
        
        self.execute(query)
        logger.info(f"Data imported from Parquet {file_path} to {table_name}")
    
    def create_index(self, table_name: str, column_names: Union[str, List[str]], 
                    index_name: Optional[str] = None):
        """
        Create index on table columns
        
        Args:
            table_name: Target table name
            column_names: Column name(s) for index
            index_name: Optional custom index name
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        
        columns_str = ", ".join(column_names)
        
        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"
        
        query = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"
        self.execute(query)
        logger.info(f"Index {index_name} created on {table_name}({columns_str})")
    
    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive table statistics
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Dictionary containing table statistics
        """
        # Basic table info
        row_count = self.get_table_count(table_name)
        columns = self.get_table_info(table_name)
        
        # Column statistics
        column_stats = {}
        for col in columns:
            col_name = col["column_name"]
            col_type = col["column_type"]
            
            if "INT" in col_type.upper() or "FLOAT" in col_type.upper() or "DOUBLE" in col_type.upper():
                # Numeric column statistics
                stats_query = f"""
                SELECT 
                    MIN({col_name}) as min_val,
                    MAX({col_name}) as max_val,
                    AVG({col_name}) as avg_val,
                    COUNT(DISTINCT {col_name}) as distinct_count,
                    COUNT({col_name}) as non_null_count
                FROM {table_name}
                """
                stats = self.fetch_one(stats_query)
                if stats:
                    column_stats[col_name] = {
                        "type": "numeric",
                        "min": stats[0],
                        "max": stats[1],
                        "avg": stats[2],
                        "distinct_count": stats[3],
                        "non_null_count": stats[4],
                        "null_count": row_count - stats[4]
                    }
            else:
                # Text/other column statistics
                stats_query = f"""
                SELECT 
                    COUNT(DISTINCT {col_name}) as distinct_count,
                    COUNT({col_name}) as non_null_count
                FROM {table_name}
                """
                stats = self.fetch_one(stats_query)
                if stats:
                    column_stats[col_name] = {
                        "type": "categorical",
                        "distinct_count": stats[0],
                        "non_null_count": stats[1],
                        "null_count": row_count - stats[1]
                    }
        
        return {
            "table_name": table_name,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
            "column_statistics": column_stats,
            "analyzed_at": datetime.now().isoformat()
        }
    
    def search_tables(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for tables and columns containing the search term
        
        Args:
            search_term: Term to search for
            
        Returns:
            List of matching tables and columns
        """
        search_term = search_term.lower()
        results = []
        
        # Search table names
        tables = self.list_tables()
        for table in tables:
            if search_term in table.lower():
                results.append({
                    "type": "table",
                    "table_name": table,
                    "match_type": "table_name",
                    "match_value": table
                })
        
        # Search column names
        for table in tables:
            columns = self.get_table_info(table)
            for col in columns:
                if search_term in col["column_name"].lower():
                    results.append({
                        "type": "column",
                        "table_name": table,
                        "column_name": col["column_name"],
                        "column_type": col["column_type"],
                        "match_type": "column_name",
                        "match_value": col["column_name"]
                    })
        
        return results
    
    def backup_database(self, backup_path: str):
        """
        Create database backup
        
        Args:
            backup_path: Path for backup file
        """
        if not self.db_path:
            raise ValueError("Cannot backup in-memory database")
        
        query = f"EXPORT DATABASE '{backup_path}'"
        self.execute(query)
        logger.info(f"Database backed up to {backup_path}")
    
    def restore_database(self, backup_path: str):
        """
        Restore database from backup
        
        Args:
            backup_path: Path to backup file
        """
        query = f"IMPORT DATABASE '{backup_path}'"
        self.execute(query)
        logger.info(f"Database restored from {backup_path}")
    
    def optimize_database(self):
        """Run database optimization operations"""
        try:
            # Analyze all tables for query optimization
            tables = self.list_tables()
            for table in tables:
                self.execute(f"ANALYZE {table}")
            
            # Run VACUUM to reclaim space
            self.execute("VACUUM")
            
            logger.info("Database optimization completed")
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            raise
    
    def get_database_size(self) -> Dict[str, Any]:
        """
        Get database size information
        
        Returns:
            Dictionary with size information
        """
        if not self.db_path:
            return {"type": "in_memory", "size": "N/A"}
        
        try:
            db_file = Path(self.db_path)
            if db_file.exists():
                size_bytes = db_file.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                return {
                    "type": "file",
                    "path": str(db_file),
                    "size_bytes": size_bytes,
                    "size_mb": round(size_mb, 2),
                    "size_human": f"{size_mb:.2f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"
                }
            else:
                return {"type": "file", "path": str(db_file), "exists": False}
        except Exception as e:
            logger.error(f"Failed to get database size: {e}")
            return {"type": "error", "error": str(e)}
    
    def execute_script(self, script_path: str):
        """
        Execute SQL script from file
        
        Args:
            script_path: Path to SQL script file
        """
        script_file = Path(script_path)
        if not script_file.exists():
            raise FileNotFoundError(f"Script file not found: {script_path}")
        
        with open(script_file, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Split script into individual statements
        statements = [stmt.strip() for stmt in script_content.split(';') if stmt.strip()]
        
        for statement in statements:
            self.execute(statement)
        
        logger.info(f"SQL script executed: {script_path}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        if hasattr(self, 'connection') and self.connection:
            self.close()


# Utility functions for common operations
def create_knowledge_base_tables(db_manager: DuckDBManager):
    """Create standard tables for knowledge base operations"""
    
    # Documents table
    documents_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "kb_name": "VARCHAR NOT NULL",
        "filename": "VARCHAR NOT NULL",
        "file_path": "VARCHAR",
        "file_size": "BIGINT",
        "file_type": "VARCHAR",
        "content": "TEXT",
        "metadata": "JSON",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("documents", documents_schema)
    
    # Chunks table for RAG
    chunks_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "document_id": "VARCHAR NOT NULL",
        "kb_name": "VARCHAR NOT NULL",
        "chunk_index": "INTEGER NOT NULL",
        "content": "TEXT NOT NULL",
        "embedding": "FLOAT[]",
        "metadata": "JSON",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("chunks", chunks_schema)
    
    # Search history table
    search_history_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "kb_name": "VARCHAR NOT NULL",
        "query": "TEXT NOT NULL",
        "results_count": "INTEGER",
        "response_time_ms": "INTEGER",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("search_history", search_history_schema)
    
    logger.info("Knowledge base tables created successfully")


def create_analytics_tables(db_manager: DuckDBManager):
    """Create tables for analytics and monitoring"""
    
    # User activities table
    activities_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "user_id": "VARCHAR",
        "activity_type": "VARCHAR NOT NULL",
        "activity_data": "JSON",
        "duration_ms": "INTEGER",
        "success": "BOOLEAN DEFAULT TRUE",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("user_activities", activities_schema)
    
    # System metrics table
    metrics_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "metric_name": "VARCHAR NOT NULL",
        "metric_value": "DOUBLE NOT NULL",
        "metric_unit": "VARCHAR",
        "tags": "JSON",
        "recorded_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("system_metrics", metrics_schema)
    
    logger.info("Analytics tables created successfully")

# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Settings Storage using DuckDB
Provides persistent storage for application settings
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger

from sirchmunk.storage.duckdb import DuckDBManager
from sirchmunk.utils.constants import DEFAULT_SIRCHMUNK_WORK_PATH


class SettingsStorage:
    """
    Manages persistent storage of application settings using DuckDB
    
    Architecture:
    - Stores UI settings, environment variables, and configuration
    - Follows Single Responsibility Principle (SRP)
    - Provides clean interface for CRUD operations
    """
    
    def __init__(self, work_path: Optional[str] = None):
        """
        Initialize Settings Storage
        
        Args:
            work_path: Base work path. If None, uses SIRCHMUNK_WORK_PATH env variable
        """
        # Get work path from env if not provided, and expand ~ in path
        if work_path is None:
            work_path = os.getenv("SIRCHMUNK_WORK_PATH", DEFAULT_SIRCHMUNK_WORK_PATH)
        
        # Create settings storage path (expand ~ and resolve to absolute path)
        self.settings_path = Path(work_path).expanduser().resolve() / ".cache" / "settings"
        self.settings_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB
        self.db_path = str(self.settings_path / "settings.db")
        self.db = DuckDBManager(db_path=self.db_path)
        
        # Create tables if not exist
        self._initialize_tables()
        
        logger.info(f"Settings storage initialized at: {self.db_path}")
    
    def _initialize_tables(self):
        """Create database tables for settings"""
        
        # Settings table (key-value store)
        settings_schema = {
            "key": "VARCHAR PRIMARY KEY",
            "value": "TEXT NOT NULL",
            "category": "VARCHAR NOT NULL",
            "updated_at": "TIMESTAMP NOT NULL",
        }
        self.db.create_table("settings", settings_schema, if_not_exists=True)
        
        # Environment variables table
        env_schema = {
            "key": "VARCHAR PRIMARY KEY",
            "value": "TEXT",
            "description": "TEXT",
            "category": "VARCHAR",
            "updated_at": "TIMESTAMP NOT NULL",
        }
        self.db.create_table("environment", env_schema, if_not_exists=True)
    
    def save_setting(self, key: str, value: Any, category: str = "general") -> bool:
        """
        Save or update a setting
        
        Args:
            key: Setting key
            value: Setting value (will be JSON serialized)
            category: Setting category (ui, llm, env, etc.)
        
        Returns:
            True if successful
        """
        try:
            # Serialize value to JSON
            value_str = json.dumps(value) if not isinstance(value, str) else value
            
            # Check if setting exists
            existing = self.db.fetch_one(
                "SELECT key FROM settings WHERE key = ?",
                [key]
            )
            
            data = {
                "key": key,
                "value": value_str,
                "category": category,
                "updated_at": datetime.now().isoformat(),
            }
            
            if existing:
                # Update existing setting
                set_clause = {k: v for k, v in data.items() if k != "key"}
                self.db.update_data(
                    "settings",
                    set_clause=set_clause,
                    where_clause="key = ?",
                    where_params=[key]
                )
                logger.debug(f"Updated setting: {key}")
            else:
                # Insert new setting
                self.db.insert_data("settings", data)
                logger.debug(f"Created new setting: {key}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save setting {key}: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value
        
        Args:
            key: Setting key
            default: Default value if not found
        
        Returns:
            Setting value (deserialized from JSON) or default
        """
        try:
            row = self.db.fetch_one(
                "SELECT value FROM settings WHERE key = ?",
                [key]
            )
            
            if row:
                value_str = row[0]
                try:
                    return json.loads(value_str)
                except json.JSONDecodeError:
                    return value_str
            
            return default
        
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            return default
    
    def get_settings_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get all settings in a category
        
        Args:
            category: Category name
        
        Returns:
            Dictionary of key-value pairs
        """
        try:
            rows = self.db.fetch_all(
                "SELECT key, value FROM settings WHERE category = ?",
                [category]
            )
            
            settings = {}
            for row in rows:
                key, value_str = row
                try:
                    settings[key] = json.loads(value_str)
                except json.JSONDecodeError:
                    settings[key] = value_str
            
            return settings
        
        except Exception as e:
            logger.error(f"Failed to get settings for category {category}: {e}")
            return {}
    
    def get_all_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all settings grouped by category
        
        Returns:
            Dictionary with categories as keys
        """
        try:
            rows = self.db.fetch_all(
                "SELECT key, value, category FROM settings"
            )
            
            settings_by_category = {}
            for row in rows:
                key, value_str, category = row
                
                if category not in settings_by_category:
                    settings_by_category[category] = {}
                
                try:
                    settings_by_category[category][key] = json.loads(value_str)
                except json.JSONDecodeError:
                    settings_by_category[category][key] = value_str
            
            return settings_by_category
        
        except Exception as e:
            logger.error(f"Failed to get all settings: {e}")
            return {}
    
    def save_env_variable(self, key: str, value: str, description: str = "", category: str = "general") -> bool:
        """
        Save or update an environment variable
        
        Args:
            key: Environment variable key
            value: Environment variable value
            description: Description of the variable
            category: Category (llm, system, etc.)
        
        Returns:
            True if successful
        """
        try:
            # Check if env var exists
            existing = self.db.fetch_one(
                "SELECT key FROM environment WHERE key = ?",
                [key]
            )
            
            data = {
                "key": key,
                "value": value,
                "description": description,
                "category": category,
                "updated_at": datetime.now().isoformat(),
            }
            
            if existing:
                # Update existing env var
                set_clause = {k: v for k, v in data.items() if k != "key"}
                self.db.update_data(
                    "environment",
                    set_clause=set_clause,
                    where_clause="key = ?",
                    where_params=[key]
                )
                logger.debug(f"Updated env var: {key}")
            else:
                # Insert new env var
                self.db.insert_data("environment", data)
                logger.debug(f"Created new env var: {key}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save env var {key}: {e}")
            return False
    
    def get_env_variable(self, key: str, default: str = "") -> str:
        """
        Get an environment variable value
        
        Args:
            key: Environment variable key
            default: Default value if not found
        
        Returns:
            Environment variable value or default
        """
        try:
            row = self.db.fetch_one(
                "SELECT value FROM environment WHERE key = ?",
                [key]
            )
            
            return row[0] if row and row[0] else default
        
        except Exception as e:
            logger.error(f"Failed to get env var {key}: {e}")
            return default
    
    def get_all_env_variables(self) -> Dict[str, Dict[str, str]]:
        """
        Get all environment variables grouped by category
        
        Returns:
            Dictionary with categories as keys
        """
        try:
            rows = self.db.fetch_all(
                "SELECT key, value, description, category FROM environment"
            )
            
            env_by_category = {}
            for row in rows:
                key, value, description, category = row
                
                if category not in env_by_category:
                    env_by_category[category] = {}
                
                env_by_category[category][key] = {
                    "value": value or "",
                    "description": description or ""
                }
            
            return env_by_category
        
        except Exception as e:
            logger.error(f"Failed to get all env vars: {e}")
            return {}
    
    def delete_setting(self, key: str) -> bool:
        """Delete a setting"""
        try:
            self.db.delete_data("settings", "key = ?", [key])
            logger.info(f"Deleted setting: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete setting {key}: {e}")
            return False
    
    def delete_env_variable(self, key: str) -> bool:
        """Delete an environment variable"""
        try:
            self.db.delete_data("environment", "key = ?", [key])
            logger.info(f"Deleted env var: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete env var {key}: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()
            logger.info("Settings storage closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        if hasattr(self, 'db') and self.db:
            self.close()

# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Knowledge Manager using DuckDB and Parquet
Manages KnowledgeCluster objects with persistence
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger

from .duckdb import DuckDBManager
from sirchmunk.schema.knowledge import (
    KnowledgeCluster,
    EvidenceUnit,
    Constraint,
    WeakSemanticEdge,
    Lifecycle,
    AbstractionLevel
)
from ..utils.constants import DEFAULT_WORK_PATH


class KnowledgeStorage:
    """
    Manages persistent storage of KnowledgeCluster objects using DuckDB and Parquet
    
    Architecture:
    - Uses KnowledgeCluster as core schema
    - Stores data in Parquet format for efficient storage and querying
    - Provides full CRUD operations with fuzzy search capabilities
    - Follows Single Responsibility Principle (SRP)
    
    Storage Path: {WORK_PATH}/.cache/knowledge/
    """
    
    def __init__(self, work_path: Optional[str] = None):
        """
        Initialize Knowledge Manager
        
        Args:
            work_path: Base work path. If None, uses WORK_PATH env variable
        """
        # Get work path from env if not provided
        if work_path is None:
            work_path = os.getenv("WORK_PATH", DEFAULT_WORK_PATH)
        
        # Create knowledge storage path
        self.knowledge_path = Path(work_path) / ".cache" / "knowledge"
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        
        # Parquet file path
        self.parquet_file = str(self.knowledge_path / "knowledge_clusters.parquet")
        
        # Initialize async lock for thread-safe parquet operations
        self._parquet_lock = asyncio.Lock()
        
        # Initialize DuckDB (in-memory for fast operations)
        self.db = DuckDBManager(db_path=None)  # In-memory database
        
        # Table name
        self.table_name = "knowledge_clusters"
        
        # Load data from parquet if exists
        self._load_from_parquet()
        
        logger.info(f"Knowledge Manager initialized at: {self.knowledge_path}")
    
    def _load_from_parquet(self):
        """Load knowledge clusters from parquet file into DuckDB"""
        try:
            if Path(self.parquet_file).exists():
                # Drop existing table first to avoid conflicts
                self.db.drop_table(self.table_name, if_exists=True)
                # Load parquet file into DuckDB table
                self.db.import_from_parquet(self.table_name, self.parquet_file, create_table=True)
                count = self.db.get_table_count(self.table_name)
                logger.info(f"Loaded {count} knowledge clusters from {self.parquet_file}")
            else:
                # Create empty table with schema
                self._create_table()
                logger.info("Created new knowledge clusters table")
        except Exception as e:
            logger.error(f"Failed to load from parquet: {e}")
            # Try to recreate table
            self.db.drop_table(self.table_name, if_exists=True)
            self._create_table()
    
    def _create_table(self):
        """Create knowledge clusters table with schema"""
        schema = {
            "id": "VARCHAR PRIMARY KEY",
            "name": "VARCHAR NOT NULL",
            "description": "VARCHAR",
            "content": "VARCHAR",
            "scripts": "VARCHAR",  # JSON array
            "resources": "VARCHAR",  # JSON array
            "evidences": "VARCHAR",  # JSON array
            "patterns": "VARCHAR",  # JSON array
            "constraints": "VARCHAR",  # JSON array
            "confidence": "DOUBLE",
            "abstraction_level": "VARCHAR",
            "landmark_potential": "DOUBLE",
            "hotness": "DOUBLE",
            "lifecycle": "VARCHAR",
            "create_time": "TIMESTAMP",
            "last_modified": "TIMESTAMP",
            "version": "INTEGER",
            "related_clusters": "VARCHAR",  # JSON array
            "search_results": "VARCHAR",  # JSON array
        }
        self.db.create_table(self.table_name, schema, if_not_exists=True)
        logger.info(f"Created table {self.table_name}")
    
    async def _save_to_parquet(self):
        """
        Save current knowledge clusters to parquet file with thread-safe atomic write.
        Uses async lock to prevent concurrent writes and temp file + rename for atomicity.
        """
        async with self._parquet_lock:
            temp_file = None
            try:
                # Generate temporary file path with timestamp for uniqueness
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                temp_file = f"{self.parquet_file}.tmp_{timestamp}"
                
                # Export table to temporary parquet file
                self.db.export_to_parquet(self.table_name, temp_file)
                
                # Verify temporary file was created successfully
                if not Path(temp_file).exists():
                    raise IOError(f"Temporary file not created: {temp_file}")
                
                # Atomically replace the target file with the temporary file
                # os.replace() is atomic on both Unix and Windows
                os.replace(temp_file, self.parquet_file)
                
                logger.debug(f"Atomically saved knowledge clusters to {self.parquet_file}")
                
            except Exception as e:
                logger.error(f"Failed to save to parquet: {e}")
                # Clean up temporary file if it exists
                if temp_file and Path(temp_file).exists():
                    try:
                        Path(temp_file).unlink()
                        logger.debug(f"Cleaned up temporary file: {temp_file}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temp file {temp_file}: {cleanup_error}")
                raise
    
    def _cluster_to_row(self, cluster: KnowledgeCluster) -> Dict[str, Any]:
        """Convert KnowledgeCluster to database row"""
        # Handle list/string fields for description and content
        description_str = (
            json.dumps(cluster.description) 
            if isinstance(cluster.description, list) 
            else cluster.description
        )
        content_str = (
            json.dumps(cluster.content) 
            if isinstance(cluster.content, list) 
            else cluster.content
        )
        
        return {
            "id": cluster.id,
            "name": cluster.name,
            "description": description_str,
            "content": content_str,
            "scripts": json.dumps(cluster.scripts) if cluster.scripts else None,
            "resources": json.dumps(cluster.resources) if cluster.resources else None,
            "evidences": json.dumps([e.to_dict() for e in cluster.evidences]),
            "patterns": json.dumps(cluster.patterns),
            "constraints": json.dumps([c.to_dict() for c in cluster.constraints]),
            "confidence": cluster.confidence,
            "abstraction_level": cluster.abstraction_level.name if cluster.abstraction_level else None,
            "landmark_potential": cluster.landmark_potential,
            "hotness": cluster.hotness,
            "lifecycle": cluster.lifecycle.name,
            "create_time": cluster.create_time.isoformat() if cluster.create_time else None,
            "last_modified": cluster.last_modified.isoformat() if cluster.last_modified else None,
            "version": cluster.version,
            "related_clusters": json.dumps([rc.to_dict() for rc in cluster.related_clusters]),
            "search_results": json.dumps(cluster.search_results) if cluster.search_results else None,
        }
    
    def _row_to_cluster(self, row: tuple) -> KnowledgeCluster:
        """Convert database row to KnowledgeCluster"""
        # Unpack row (order matches schema). Older tables may not include search_results.
        if len(row) == 19:
            (
                id, name, description, content, scripts, resources, evidences, patterns,
                constraints, confidence, abstraction_level, landmark_potential, hotness,
                lifecycle, create_time, last_modified, version, related_clusters, search_results
            ) = row
        elif len(row) == 18:
            (
                id, name, description, content, scripts, resources, evidences, patterns,
                constraints, confidence, abstraction_level, landmark_potential, hotness,
                lifecycle, create_time, last_modified, version, related_clusters
            ) = row
            search_results = None
        elif len(row) == 17:
            (
                id, name, description, content, scripts, resources, evidences, patterns,
                constraints, confidence, abstraction_level, landmark_potential, hotness,
                lifecycle, create_time, last_modified, version
            ) = row
            related_clusters = None
            search_results = None
        else:
            raise ValueError(f"Unexpected knowledge_clusters row length: {len(row)}")
        
        # Parse JSON fields
        try:
            description_parsed = json.loads(description) if description and description.startswith('[') else description
        except:
            description_parsed = description
        
        try:
            content_parsed = json.loads(content) if content and content.startswith('[') else content
        except:
            content_parsed = content
        
        scripts_parsed = json.loads(scripts) if scripts else None
        resources_parsed = json.loads(resources) if resources else None
        patterns_parsed = json.loads(patterns) if patterns else []
        
        # Parse evidences
        evidences_parsed = []
        if evidences:
            evidences_data = json.loads(evidences)
            for ev_dict in evidences_data:
                evidences_parsed.append(EvidenceUnit(
                    doc_id=ev_dict["doc_id"],
                    file_or_url=Path(ev_dict["file_or_url"]),
                    summary=ev_dict["summary"],
                    is_found=ev_dict["is_found"],
                    snippets=ev_dict["snippets"],
                    extracted_at=datetime.fromisoformat(ev_dict["extracted_at"]),
                    conflict_group=ev_dict.get("conflict_group")
                ))
        
        # Parse constraints
        constraints_parsed = []
        if constraints:
            constraints_data = json.loads(constraints)
            for c_dict in constraints_data:
                constraints_parsed.append(Constraint.from_dict(c_dict))
        
        # Parse related clusters
        related_clusters_parsed = []
        if related_clusters:
            related_data = json.loads(related_clusters)
            for rc_dict in related_data:
                related_clusters_parsed.append(WeakSemanticEdge.from_dict(rc_dict))
        
        # Parse search results
        search_results_parsed = []
        if search_results:
            search_results_parsed = json.loads(search_results)
        
        return KnowledgeCluster(
            id=id,
            name=name,
            description=description_parsed,
            content=content_parsed,
            scripts=scripts_parsed,
            resources=resources_parsed,
            evidences=evidences_parsed,
            patterns=patterns_parsed,
            constraints=constraints_parsed,
            confidence=confidence,
            abstraction_level=AbstractionLevel[abstraction_level] if abstraction_level else None,
            landmark_potential=landmark_potential,
            hotness=hotness,
            lifecycle=Lifecycle[lifecycle],
            create_time=datetime.fromisoformat(create_time) if create_time else None,
            last_modified=datetime.fromisoformat(last_modified) if last_modified else None,
            version=version,
            related_clusters=related_clusters_parsed,
            search_results=search_results_parsed,
        )
    
    async def get(self, cluster_id: str) -> Optional[KnowledgeCluster]:
        """
        Get a knowledge cluster by ID (exact match)
        
        Args:
            cluster_id: Unique cluster ID
        
        Returns:
            KnowledgeCluster if found, None otherwise
        """
        try:
            row = self.db.fetch_one(
                f"SELECT * FROM {self.table_name} WHERE id = ?",
                [cluster_id]
            )
            
            if row:
                return self._row_to_cluster(row)
            return None
        
        except Exception as e:
            logger.error(f"Failed to get cluster {cluster_id}: {e}")
            return None
    
    async def insert(self, cluster: KnowledgeCluster) -> bool:
        """
        Insert a new knowledge cluster
        
        Args:
            cluster: KnowledgeCluster to insert
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if cluster already exists
            existing = await self.get(cluster.id)
            if existing:
                logger.warning(f"Cluster {cluster.id} already exists, use update() instead")
                return False
            
            # Set creation and modification times if not set
            if not cluster.create_time:
                cluster.create_time = datetime.now()
            if not cluster.last_modified:
                cluster.last_modified = datetime.now()
            if cluster.version is None:
                cluster.version = 1
            
            # Insert into database
            row = self._cluster_to_row(cluster)
            self.db.insert_data(self.table_name, row)
            
            # Save to parquet with atomic write
            await self._save_to_parquet()
            
            logger.info(f"Inserted cluster: {cluster.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to insert cluster {cluster.id}: {e}")
            return False
    
    async def update(self, cluster: KnowledgeCluster) -> bool:
        """
        Update an existing knowledge cluster
        
        Args:
            cluster: KnowledgeCluster with updated data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if cluster exists
            existing = await self.get(cluster.id)
            if not existing:
                logger.warning(f"Cluster {cluster.id} does not exist, use insert() instead")
                return False
            
            # Update modification time and version
            cluster.last_modified = datetime.now()
            cluster.version = (cluster.version or 0) + 1
            
            # Prepare update data
            row = self._cluster_to_row(cluster)
            set_clause = {k: v for k, v in row.items() if k != "id"}
            
            # Update in database
            self.db.update_data(
                self.table_name,
                set_clause=set_clause,
                where_clause="id = ?",
                where_params=[cluster.id]
            )
            
            # Save to parquet with atomic write
            await self._save_to_parquet()
            
            logger.info(f"Updated cluster: {cluster.id} (version {cluster.version})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update cluster {cluster.id}: {e}")
            return False
    
    async def remove(self, cluster_id: str) -> bool:
        """
        Remove a knowledge cluster by ID
        
        Args:
            cluster_id: Unique cluster ID
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if cluster exists
            existing = await self.get(cluster_id)
            if not existing:
                logger.warning(f"Cluster {cluster_id} does not exist")
                return False
            
            # Delete from database
            self.db.delete_data(self.table_name, "id = ?", [cluster_id])
            
            # Save to parquet with atomic write
            await self._save_to_parquet()
            
            logger.info(f"Removed cluster: {cluster_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to remove cluster {cluster_id}: {e}")
            return False
    
    async def clear(self) -> bool:
        """
        Clear all knowledge clusters
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Drop and recreate table
            self.db.drop_table(self.table_name, if_exists=True)
            self._create_table()
            
            # Delete parquet file
            if Path(self.parquet_file).exists():
                Path(self.parquet_file).unlink()
            
            logger.info("Cleared all knowledge clusters")
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear knowledge clusters: {e}")
            return False
    
    async def find(self, query: str, limit: int = 10) -> List[KnowledgeCluster]:
        """
        Find knowledge clusters using fuzzy search
        Searches in: id, name, description, content, patterns
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        
        Returns:
            List of matching KnowledgeCluster objects
        """
        try:
            # Fuzzy search using LIKE with wildcards
            search_pattern = f"%{query}%"
            
            sql = f"""
            SELECT * FROM {self.table_name}
            WHERE 
                id LIKE ? OR
                name LIKE ? OR
                description LIKE ? OR
                content LIKE ? OR
                patterns LIKE ?
            ORDER BY 
                CASE 
                    WHEN id = ? THEN 1
                    WHEN name LIKE ? THEN 2
                    WHEN description LIKE ? THEN 3
                    ELSE 4
                END
            LIMIT ?
            """
            
            params = [
                search_pattern,  # id LIKE
                search_pattern,  # name LIKE
                search_pattern,  # description LIKE
                search_pattern,  # content LIKE
                search_pattern,  # patterns LIKE
                query,           # exact id match
                f"{query}%",     # name starts with
                f"%{query}%",    # description contains
                limit
            ]
            
            rows = self.db.fetch_all(sql, params)
            
            clusters = [self._row_to_cluster(row) for row in rows]
            
            logger.debug(f"Found {len(clusters)} clusters matching '{query}'")
            return clusters
        
        except Exception as e:
            logger.error(f"Failed to search clusters with query '{query}': {e}")
            return []
    
    async def merge(self, clusters: List[KnowledgeCluster]) -> Optional[KnowledgeCluster]:
        """
        Merge multiple knowledge clusters into one
        
        Strategy:
        - Use first cluster as base
        - Merge evidences, patterns, constraints from all clusters
        - Average numeric scores (confidence, hotness, etc.)
        - Update version and timestamps
        
        Args:
            clusters: List of KnowledgeCluster objects to merge
        
        Returns:
            Merged KnowledgeCluster, or None if merge fails
        """
        if not clusters:
            logger.warning("No clusters to merge")
            return None
        
        if len(clusters) == 1:
            logger.warning("Only one cluster provided, returning as-is")
            return clusters[0]
        
        try:
            # Use first cluster as base
            merged = clusters[0]
            
            # Merge content and descriptions
            all_descriptions = []
            all_contents = []
            
            for cluster in clusters:
                # Handle descriptions
                if isinstance(cluster.description, list):
                    all_descriptions.extend(cluster.description)
                else:
                    all_descriptions.append(cluster.description)
                
                # Handle contents
                if isinstance(cluster.content, list):
                    all_contents.extend(cluster.content)
                else:
                    all_contents.append(cluster.content)
            
            merged.description = list(set(all_descriptions))  # Deduplicate
            merged.content = list(set(all_contents))  # Deduplicate
            
            # Merge evidences (deduplicate by doc_id)
            evidences_map = {}
            for cluster in clusters:
                for evidence in cluster.evidences:
                    if evidence.doc_id not in evidences_map:
                        evidences_map[evidence.doc_id] = evidence
            merged.evidences = list(evidences_map.values())
            
            # Merge patterns (deduplicate)
            all_patterns = []
            for cluster in clusters:
                all_patterns.extend(cluster.patterns)
            merged.patterns = list(set(all_patterns))
            
            # Merge constraints (deduplicate by condition)
            constraints_map = {}
            for cluster in clusters:
                for constraint in cluster.constraints:
                    if constraint.condition not in constraints_map:
                        constraints_map[constraint.condition] = constraint
            merged.constraints = list(constraints_map.values())
            
            # Merge related clusters (deduplicate by target_cluster_id)
            related_map = {}
            for cluster in clusters:
                for related in cluster.related_clusters:
                    if related.target_cluster_id not in related_map:
                        related_map[related.target_cluster_id] = related
                    else:
                        # Average weights if duplicate
                        existing = related_map[related.target_cluster_id]
                        existing.weight = (existing.weight + related.weight) / 2
            merged.related_clusters = list(related_map.values())
            
            # Average numeric scores
            valid_confidences = [c.confidence for c in clusters if c.confidence is not None]
            if valid_confidences:
                merged.confidence = sum(valid_confidences) / len(valid_confidences)
            
            valid_hotness = [c.hotness for c in clusters if c.hotness is not None]
            if valid_hotness:
                merged.hotness = sum(valid_hotness) / len(valid_hotness)
            
            valid_landmark = [c.landmark_potential for c in clusters if c.landmark_potential is not None]
            if valid_landmark:
                merged.landmark_potential = sum(valid_landmark) / len(valid_landmark)
            
            # Update metadata
            merged.name = f"{merged.name} (merged)"
            merged.last_modified = datetime.now()
            merged.version = (merged.version or 0) + 1
            
            # Update the merged cluster in database
            await self.update(merged)
            
            # Remove source clusters (except the first one which is now merged)
            for cluster in clusters[1:]:
                await self.remove(cluster.id)
            
            logger.info(f"Merged {len(clusters)} clusters into {merged.id}")
            return merged
        
        except Exception as e:
            logger.error(f"Failed to merge clusters: {e}")
            return None
    
    async def split(self, cluster: KnowledgeCluster, num_splits: int = 2) -> List[KnowledgeCluster]:
        """
        Split a knowledge cluster into multiple smaller clusters
        
        Strategy:
        - Split evidences evenly across new clusters
        - Distribute patterns and constraints
        - Create new cluster IDs based on original ID
        
        Args:
            cluster: KnowledgeCluster to split
            num_splits: Number of clusters to split into (default: 2)
        
        Returns:
            List of new KnowledgeCluster objects
        """
        if num_splits < 2:
            logger.warning("num_splits must be >= 2, returning original cluster")
            return [cluster]
        
        try:
            new_clusters = []
            
            # Split evidences
            evidences_per_cluster = len(cluster.evidences) // num_splits
            if evidences_per_cluster == 0:
                logger.warning("Not enough evidences to split, returning original cluster")
                return [cluster]
            
            for i in range(num_splits):
                # Create new cluster ID
                new_id = f"{cluster.id}_split{i+1}"
                
                # Calculate evidence range
                start_idx = i * evidences_per_cluster
                end_idx = start_idx + evidences_per_cluster if i < num_splits - 1 else len(cluster.evidences)
                
                # Create new cluster
                new_cluster = KnowledgeCluster(
                    id=new_id,
                    name=f"{cluster.name} (part {i+1})",
                    description=cluster.description,
                    content=cluster.content,
                    scripts=cluster.scripts,
                    resources=cluster.resources,
                    evidences=cluster.evidences[start_idx:end_idx],
                    patterns=cluster.patterns[i::num_splits],  # Distribute patterns
                    constraints=cluster.constraints[i::num_splits],  # Distribute constraints
                    confidence=cluster.confidence,
                    abstraction_level=cluster.abstraction_level,
                    landmark_potential=cluster.landmark_potential,
                    hotness=cluster.hotness,
                    lifecycle=Lifecycle.EMERGING,  # New clusters are emerging
                    create_time=datetime.now(),
                    last_modified=datetime.now(),
                    version=1,
                    related_clusters=cluster.related_clusters,
                )
                
                # Insert new cluster
                await self.insert(new_cluster)
                new_clusters.append(new_cluster)
            
            # Remove original cluster
            await self.remove(cluster.id)
            
            logger.info(f"Split cluster {cluster.id} into {num_splits} clusters")
            return new_clusters
        
        except Exception as e:
            logger.error(f"Failed to split cluster {cluster.id}: {e}")
            return [cluster]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored knowledge clusters
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = self.db.analyze_table(self.table_name)
            
            # Add custom stats
            total_count = self.db.get_table_count(self.table_name)
            
            # Count by lifecycle
            lifecycle_counts = {}
            for lifecycle in Lifecycle:
                count_row = self.db.fetch_one(
                    f"SELECT COUNT(*) FROM {self.table_name} WHERE lifecycle = ?",
                    [lifecycle.name]
                )
                lifecycle_counts[lifecycle.name] = count_row[0] if count_row else 0
            
            # Average confidence
            avg_confidence_row = self.db.fetch_one(
                f"SELECT AVG(confidence) FROM {self.table_name} WHERE confidence IS NOT NULL"
            )
            avg_confidence = avg_confidence_row[0] if avg_confidence_row and avg_confidence_row[0] else 0
            
            stats["custom_stats"] = {
                "total_clusters": total_count,
                "lifecycle_distribution": lifecycle_counts,
                "average_confidence": round(avg_confidence, 4) if avg_confidence else None,
                "parquet_file": self.parquet_file,
                "parquet_exists": Path(self.parquet_file).exists(),
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()
            logger.info("Knowledge Manager closed")
    
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

# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Knowledge Base API endpoints
Provides CRUD and analytics for KnowledgeCluster objects
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel

from sirchmunk.storage.knowledge_manager import KnowledgeManager
from sirchmunk.schema.knowledge import AbstractionLevel

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

# Initialize Knowledge Manager
km = KnowledgeManager()

# === Request/Response Models ===

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

# === API Endpoints ===

@router.get("/list")
async def list_knowledge_bases_alias():
    """Alias for /clusters endpoint (backward compatibility)"""
    return await get_all_clusters(limit=100)


@router.get("/clusters")
async def get_all_clusters(
    limit: int = 100,
    lifecycle: Optional[str] = None,
    abstraction_level: Optional[str] = None
):
    """
    Get all knowledge clusters with optional filtering
    
    Query params:
        limit: Maximum number of clusters to return
        lifecycle: Filter by lifecycle (STABLE, EMERGING, CONTESTED, DEPRECATED)
        abstraction_level: Filter by abstraction level
    """
    try:
        stats = km.get_stats()
        
        # Get all clusters by searching with empty query (returns all)
        all_clusters = []
        
        # Fetch clusters using DuckDB directly
        sql = "SELECT * FROM knowledge_clusters"
        where_clauses = []
        params = []
        
        if lifecycle:
            where_clauses.append("lifecycle = ?")
            params.append(lifecycle.upper())
        
        if abstraction_level:
            where_clauses.append("abstraction_level = ?")
            params.append(abstraction_level.upper())
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        sql += f" ORDER BY last_modified DESC LIMIT {limit}"
        
        try:
            rows = km.db.fetch_all(sql, params if params else None)
        except Exception as fetch_error:
            # If table is missing or schema is out of date, recreate and return empty list.
            km._create_table()
            rows = []
        clusters = [km._row_to_cluster(row) for row in rows]
        
        return {
            "success": True,
                "count": len(clusters),
                "total": stats.get('custom_stats', {}).get('total_clusters', 0),
                "data": [c.to_dict() for c in clusters]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clusters/{cluster_id}")
async def get_cluster(cluster_id: str):
    """Get a specific knowledge cluster by ID"""
    try:
        cluster = await km.get(cluster_id)
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")
    
        return {
            "success": True,
                "data": cluster.to_dict()
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_clusters(request: SearchRequest):
    """
    Search knowledge clusters by query
    
    Searches across: id, name, description, content, patterns
    """
    try:
        results = await km.find(request.query, limit=request.limit)
    
        return {
            "success": True,
                "query": request.query,
                "count": len(results),
                "data": [c.to_dict() for c in results]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_knowledge_stats():
    """
    Get comprehensive knowledge base statistics
    
    Returns:
        - Total clusters
        - Lifecycle distribution
        - Abstraction level distribution
        - Average confidence
        - Hotness distribution
        - Top patterns
        - Recent activity
    """
    try:
        stats = km.get_stats()
        custom_stats = stats.get('custom_stats', {})
        
        # Get lifecycle distribution
        lifecycle_dist = custom_stats.get('lifecycle_distribution', {})
        
        # Get abstraction level distribution
        abstraction_dist = {}
        for level in AbstractionLevel:
            count_row = km.db.fetch_one(
                "SELECT COUNT(*) FROM knowledge_clusters WHERE abstraction_level = ?",
                [level.name]
            )
            abstraction_dist[level.name] = count_row[0] if count_row else 0
        
        # Get confidence statistics
        confidence_stats_row = km.db.fetch_one(
            """
            SELECT 
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence,
                AVG(confidence) as avg_confidence
            FROM knowledge_clusters 
            WHERE confidence IS NOT NULL
            """
        )
        
        confidence_stats = {
            "min": confidence_stats_row[0] if confidence_stats_row and confidence_stats_row[0] else 0,
            "max": confidence_stats_row[1] if confidence_stats_row and confidence_stats_row[1] else 0,
            "avg": round(confidence_stats_row[2], 4) if confidence_stats_row and confidence_stats_row[2] else 0,
        }
        
        # Get hotness statistics
        hotness_stats_row = km.db.fetch_one(
            """
            SELECT 
                MIN(hotness) as min_hotness,
                MAX(hotness) as max_hotness,
                AVG(hotness) as avg_hotness
            FROM knowledge_clusters 
            WHERE hotness IS NOT NULL
            """
        )
        
        hotness_stats = {
            "min": hotness_stats_row[0] if hotness_stats_row and hotness_stats_row[0] else 0,
            "max": hotness_stats_row[1] if hotness_stats_row and hotness_stats_row[1] else 0,
            "avg": round(hotness_stats_row[2], 4) if hotness_stats_row and hotness_stats_row[2] else 0,
        }
        
        # Get top 10 most recent clusters
        recent_rows = km.db.fetch_all(
            """
            SELECT id, name, last_modified 
            FROM knowledge_clusters 
            ORDER BY last_modified DESC 
            LIMIT 10
            """
        )
        
        recent_clusters = [
            {
                "id": row[0],
                "name": row[1],
                "last_modified": row[2]
            }
            for row in recent_rows
        ]
        
        # Get top 10 highest confidence clusters
        top_confidence_rows = km.db.fetch_all(
            """
            SELECT id, name, confidence 
            FROM knowledge_clusters 
            WHERE confidence IS NOT NULL
            ORDER BY confidence DESC 
            LIMIT 10
            """
        )
        
        top_confidence = [
            {
                "id": row[0],
                "name": row[1],
                "confidence": row[2]
            }
            for row in top_confidence_rows
        ]
        
        # Get top 10 hottest clusters
        top_hotness_rows = km.db.fetch_all(
            """
            SELECT id, name, hotness 
            FROM knowledge_clusters 
            WHERE hotness IS NOT NULL
            ORDER BY hotness DESC 
            LIMIT 10
            """
        )
        
        top_hotness = [
            {
                "id": row[0],
                "name": row[1],
                "hotness": row[2]
            }
            for row in top_hotness_rows
        ]
        
        # Timeline data (clusters created per day for last 30 days)
        timeline_rows = km.db.fetch_all(
            """
            SELECT 
                CAST(create_time AS DATE) as date,
                COUNT(*) as count
            FROM knowledge_clusters
            WHERE create_time >= current_date - INTERVAL '30 days'
            GROUP BY CAST(create_time AS DATE)
            ORDER BY date ASC
            """
        )
        
        timeline = [
            {
                "date": str(row[0]),
                "count": row[1]
            }
            for row in timeline_rows
        ]
    
        return {
            "success": True,
            "data": {
                    "overview": {
                        "total_clusters": custom_stats.get('total_clusters', 0),
                        "avg_confidence": custom_stats.get('average_confidence', 0),
                    },
                    "lifecycle_distribution": lifecycle_dist,
                    "abstraction_level_distribution": abstraction_dist,
                    "confidence_stats": confidence_stats,
                    "hotness_stats": hotness_stats,
                    "recent_clusters": recent_clusters,
                    "top_confidence_clusters": top_confidence,
                    "top_hotness_clusters": top_hotness,
                    "timeline": timeline,
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patterns")
async def get_top_patterns(limit: int = 20):
    """
    Get most common patterns across all clusters
    
    Query params:
        limit: Number of top patterns to return
    """
    try:
        # Fetch all patterns and count occurrences
        rows = km.db.fetch_all("SELECT patterns FROM knowledge_clusters WHERE patterns IS NOT NULL")
        
        import json
        from collections import Counter
        
        pattern_counter = Counter()
        for row in rows:
            patterns_json = row[0]
            if patterns_json:
                patterns = json.loads(patterns_json)
                pattern_counter.update(patterns)
        
        top_patterns = [
            {"pattern": pattern, "count": count}
            for pattern, count in pattern_counter.most_common(limit)
        ]
    
        return {
            "success": True,
                "count": len(top_patterns),
                "data": top_patterns
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph")
async def get_knowledge_graph():
    """
    Get knowledge graph data (nodes and edges)
    
    Returns clusters as nodes and related_clusters as edges
    """
    try:
        # Get all clusters
        rows = km.db.fetch_all(
            "SELECT id, name, confidence, hotness, lifecycle, abstraction_level, related_clusters FROM knowledge_clusters"
        )
        
        import json
        
        nodes = []
        edges = []
        
        for row in rows:
            cluster_id, name, confidence, hotness, lifecycle, abstraction_level, related_clusters_json = row
            
            # Add node
            nodes.append({
                "id": cluster_id,
                "name": name,
                "confidence": confidence,
                "hotness": hotness,
                "lifecycle": lifecycle,
                "abstraction_level": abstraction_level,
            })
            
            # Add edges
            if related_clusters_json:
                related_clusters = json.loads(related_clusters_json)
                for rc in related_clusters:
                    edges.append({
                        "source": cluster_id,
                        "target": rc["target_cluster_id"],
                        "weight": rc["weight"],
                        "type": rc["source"]
                    })
    
        return {
            "success": True,
                "data": {
                    "nodes": nodes,
                    "edges": edges
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clusters/{cluster_id}")
async def delete_cluster(cluster_id: str):
    """Delete a knowledge cluster"""
    try:
        success = await km.remove(cluster_id)
        if not success:
            raise HTTPException(status_code=404, detail="Cluster not found")
    
        return {
            "success": True,
                "message": f"Cluster {cluster_id} deleted successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clusters")
async def clear_all_clusters():
    """Clear all knowledge clusters (use with caution!)"""
    try:
        success = await km.clear()
        
        return {
            "success": success,
            "message": "All clusters cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

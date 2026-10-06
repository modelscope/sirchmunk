# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Knowledge cluster evolver — organize knowledge clusters at runtime.

Periodically connects, merges, refreshes, and organises KnowledgeClusters
that are created at search time, so the knowledge graph improves with use.

The evolver is invoked via ``KnowledgeEvolver.step(cluster)`` after each
``AgenticSearch.search()`` call. It checks four trigger conditions and
executes the corresponding phase:

* Phase 1: connect & merge (every step, when new cluster count >= _EVO_CONNECT_MERGE_COUNT)
* Phase 2: edge refresh (every step, when refresh count >= _EVO_EDGE_REFRESH_COUNT)
* Phase 3: meta cluster detection (>= _EVO_STEP_INTERVAL steps since last run,
and >= _EVO_META_DETECTION_COUNT change in non-meta cluster count)
* Phase 4: global update (>= _EVO_STEP_INTERVAL steps since last run, 
and >= _EVO_GLOBAL_UPDATE_COUNT change in non-meta cluster count)
"""

import asyncio
import json
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import igraph as ig
    import leidenalg
except ImportError:  # Optional: only required for meta-cluster detection.
    ig = None
    leidenalg = None

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import (
    EVO_META_QUERY,
    EVO_REFINE_CLUSTER
)
from sirchmunk.schema.knowledge import (
    AbstractionLevel,
    KnowledgeCluster,
    Lifecycle,
    WeakSemanticEdge
)
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.storage.lifecycle import LifecycleError, get_runtime, write_json_atomic
from sirchmunk.utils.embedding_util import EmbeddingUtil, compute_text_hash
from sirchmunk.utils import LogCallback, create_logger

# Minimum number of new clusters required to trigger connect and merge phase
_EVO_CONNECT_MERGE_COUNT = 5

# Similarity threshold for connecting clusters through weak semantic edges
_EVO_CONNECT_SIMILARITY_THRESHOLD = 0.60

# Similarity threshold for merging clusters
_EVO_MERGE_SIMILARITY_THRESHOLD = 0.90

# Minimum number of edge refreshes required to trigger edge refresh phase
_EVO_EDGE_REFRESH_COUNT = 20

# Maximum number of clusters to consider for edge refresh in a single step
_EVO_EDGE_REFRESH_TOPK = 5

# Minimum number of clusters required to trigger meta cluster detection phase
_EVO_META_DETECTION_COUNT = 100

# Maximum number of queries to consider for building meta cluster query
_EVO_META_DETECTION_MAX_QUERIES = 50

# Minimum number of clusters required to trigger global update phase
_EVO_GLOBAL_UPDATE_COUNT = 500

# Similarity threshold for disconnecting clusters during global update phase
_EVO_GLOBAL_UPDATE_DISCONNECT_SIMILARITY_THRESHOLD = 0.30

# Threshold for low hotness and confidence to trigger global update phase
_EVO_GLOBAL_UPDATE_HOTNESS_THRESHOLD = 0.1

# Threshold for low hotness and confidence to trigger global update phase
_EVO_GLOBAL_UPDATE_CONFIDENCE_THRESHOLD = 0.3

# Refined hotness value for clusters after global update phase
_EVO_GLOBAL_UPDATE_REFINED_HOTNESS = 0.3

# Refined confidence value for clusters after global update phase
_EVO_GLOBAL_UPDATE_REFINED_CONFIDENCE = 0.5

# Maximum number of clusters to consider for global update in a single step
_EVO_GLOBAL_UPDATE_MAX_CLUSTERS = 20

# Maximum number of concurrent LLM calls during evolver phases
_EVO_LLM_MAX_CONCURRENCY = 5

# Step interval for meta cluster detection and global update phases
_EVO_STEP_INTERVAL = 10


@dataclass
class EvolveManifest:
    """Tracks the state of the knowledge evolver for resume and incremental processing."""

    version: str = "1.0"
    current_step: int = 0
    cluster_ids_buffer: List[str] = field(default_factory=list)
    last_evolve_at: Optional[str] = None

    connect_merge_cluster_ids: List[str] = field(default_factory=list)
    edge_refresh_cluster_ids: List[str] = field(default_factory=list)

    last_meta_detection_step: int = 0
    last_meta_detection_cluster_count: int = 0
    last_global_update_step: int = 0
    last_global_update_cluster_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str, *, strict: bool = False) -> "EvolveManifest":
        """Read v1.0 without migration; strict mode validates the complete format.

        Lists may contain repeated IDs (multiple requests can enqueue one cluster).
        The permissive default retains the standalone SDK's legacy behaviour.
        """
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate evolve manifest field")
                result[key] = value
            return result

        def invalid_constant(_value):
            raise ValueError("Non-finite evolve manifest value")

        if strict:
            data = json.loads(json_str, object_pairs_hook=unique_object,
                              parse_constant=invalid_constant)
            if not isinstance(data, dict) or set(data) != {item.name for item in fields(cls)}:
                raise ValueError("Invalid evolve manifest field set")
            if data["version"] != "1.0":
                raise ValueError("Unsupported evolve manifest version")
            for item in fields(cls):
                value = data[item.name]
                if item.default_factory is list:
                    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                        raise ValueError(f"Invalid evolve manifest list: {item.name}")
                elif type(item.default) is int:
                    if type(value) is not int or value < 0:
                        raise ValueError(f"Invalid evolve manifest counter: {item.name}")
                elif item.name == "last_evolve_at" and value is not None and not isinstance(value, str):
                    raise ValueError("Invalid evolve manifest timestamp type")
            return cls(**data)
        data = json.loads(json_str)
        return cls(
            version=data.get("version", "1.0"),
            current_step=data.get("current_step", 0),
            cluster_ids_buffer=data.get("cluster_ids_buffer", []),
            last_evolve_at=data.get("last_evolve_at"),
            connect_merge_cluster_ids=data.get("connect_merge_cluster_ids", []),
            edge_refresh_cluster_ids=data.get("edge_refresh_cluster_ids", []),
            last_meta_detection_step=data.get("last_meta_detection_step", 0),
            last_meta_detection_cluster_count=data.get("last_meta_detection_cluster_count", 0),
            last_global_update_step=data.get("last_global_update_step", 0),
            last_global_update_cluster_count=data.get("last_global_update_cluster_count", 0)
        )


def _managed_failure(method):
    """Never report a partially applied managed evolution as successful."""
    @wraps(method)
    async def checked(self, *args, **kwargs):
        try:
            return await method(self, *args, **kwargs)
        except BaseException:
            if self._runtime is not None:
                self._runtime.fail("evolution_failed")
            raise
    return checked


class KnowledgeEvolver:
    """Periodically connects, merges, refreshes, and organises KnowledgeClusters at runtime."""

    def __init__(
        self,
        llm: OpenAIChat,
        embedding: Optional[EmbeddingUtil],
        knowledge_storage: KnowledgeStorage,
        work_path: Union[str, Path],
        log_callback: LogCallback = None,
    ):
        if embedding is None:
            raise ValueError("EmbeddingUtil instance is required for KnowledgeEvolver.")

        self._llm = llm
        self._embedding = embedding
        self._storage = knowledge_storage
        self._work_path = Path(work_path).expanduser().resolve()
        self._runtime = get_runtime(self._work_path)
        self._log = create_logger(log_callback=log_callback)

        self._evolve_dir = self._work_path / ".cache" / "evolve"
        self._manifest_path = self._evolve_dir / "manifest.json"
        if self._runtime is not None:
            # Requests own their LLM/logging, but the root owns the authoritative
            # manifest and phase lock. Do not reload a second stale copy.
            with self._runtime.io_lock:
                self._ensure_writable()
                try:
                    self._evolve_dir.mkdir(parents=True, exist_ok=True)
                    if not hasattr(self._runtime, "_evolve_manifest"):
                        manifest = self._load_manifest()
                        self._runtime._evolve_manifest = manifest
                    if not hasattr(self._runtime, "_evolve_lock"):
                        self._runtime._evolve_lock = asyncio.Lock()
                    self._manifest = self._runtime._evolve_manifest
                    self._evolve_lock = self._runtime._evolve_lock
                except BaseException:
                    self._runtime.fail("evolve_manifest_initialization_failed")
                    raise
        else:
            self._evolve_dir.mkdir(parents=True, exist_ok=True)
            self._manifest = self._load_manifest()
            self._evolve_lock = asyncio.Lock()
        
        # A semaphore to limit the concurrent LLM calls during evolver phases
        self._llm_semaphore = asyncio.Semaphore(_EVO_LLM_MAX_CONCURRENCY)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    @_managed_failure
    async def step(self, cluster: KnowledgeCluster):
        """
        Step the evolver with locking to ensure only one step is processed at a time.

        Args:
            cluster (KnowledgeCluster): The new knowledge cluster to process.
        """
        with self._manifest_guard():
            self._ensure_writable()
            self._manifest.current_step += 1
            self._manifest.cluster_ids_buffer.append(cluster.id)
            self._save_manifest(self._manifest)

        if self._evolve_lock.locked():
            await self._log.info(
                "[Evolve] Evolve step is already in progress. "
                "The new cluster has been added to the buffer for later processing."
            )
            return

        async with self._evolve_lock:
            self._ensure_writable()
            await self._log.info(
                "[Evolve] Starting evolver step with "
                f"{len(self._manifest.cluster_ids_buffer)} clusters in the buffer."
            )
            await self._step()
            await self._log.info("[Evolve] Evolver step completed.")

    @_managed_failure
    async def _step(self):
        """Step the evolver to process the clusters in the buffer."""
        # Deduplicate and clear the buffer after processing
        with self._manifest_guard():
            self._ensure_writable()
            cluster_ids = list(set(self._manifest.cluster_ids_buffer))
            buffered_count = len(self._manifest.cluster_ids_buffer)
            if self._runtime is None:
                self._manifest.cluster_ids_buffer.clear()

        new_ids, refreshed_ids = [], []

        # Check the cluster is created or reused
        # created: | last_modified - created | < 1 second
        for cluster_id in cluster_ids:
            cluster = await self._storage.get(cluster_id)
            if cluster is None:
                await self._log.warning(
                    f"[Evolve] Cluster {cluster_id} not found in storage. Skipping."
                )
                continue

            if abs((cluster.last_modified - cluster.create_time).total_seconds()) < 1:
                # This cluster is newly created
                if self._runtime is None:
                    self._manifest.connect_merge_cluster_ids.append(cluster.id)
                else:
                    new_ids.append(cluster.id)
            else:
                # This cluster is reused with embedding refresh
                if self._runtime is None:
                    self._manifest.edge_refresh_cluster_ids.append(cluster.id)
                else:
                    refreshed_ids.append(cluster.id)

        if self._runtime is not None:
            # Keep the captured prefix durable during awaits. Arrivals can save
            # the shared manifest while classification is still in progress.
            # Move only that prefix to the phase queues, then persist before IO.
            with self._runtime.io_lock:
                self._manifest.connect_merge_cluster_ids.extend(new_ids)
                self._manifest.edge_refresh_cluster_ids.extend(refreshed_ids)
                del self._manifest.cluster_ids_buffer[:buffered_count]
                self._save_manifest(self._manifest)

        # Check and execute phases based on triggers
        has_evolved = False

        try:
            # Phase 1: Connect and Merge
            if self._should_connect_merge():
                await self._log.info(
                    f"[Evolve] Phase 1: Connect and Merge triggered "
                    f"at step {self._manifest.current_step}."
                )
                await self._connect_and_merge()
                has_evolved = True

            # Phase 2: Refresh Edges
            if self._should_refresh_edges():
                await self._log.info(
                    f"[Evolve] Phase 2: Refresh Edges triggered "
                    f"at step {self._manifest.current_step}."
                )
                await self._refresh_edges()
                has_evolved = True

            # Phase 3: Detect Meta Clusters
            if self._should_detect_meta_clusters():
                await self._log.info(
                    f"[Evolve] Phase 3: Detect Meta Clusters triggered "
                    f"at step {self._manifest.current_step}."
                )
                await self._detect_meta_clusters()
                has_evolved = True

            # Phase 4: Global Update
            if self._should_global_update():
                await self._log.info(
                    f"[Evolve] Phase 4: Global Update triggered "
                    f"at step {self._manifest.current_step}."
                )
                await self._global_update()
                has_evolved = True

        except Exception as e:
            if self._runtime is not None:
                raise
            await self._log.error(f"[Evolve] Error during evolve step: {e}")
        finally:
            # Legacy mode still checkpoints partial progress on failure. Managed
            # mode only commits below after every triggered phase has succeeded.
            if self._runtime is None:
                if has_evolved:
                    self._manifest.last_evolve_at = datetime.now(timezone.utc).isoformat()
                self._save_manifest(self._manifest)
        if self._runtime is not None:
            with self._runtime.io_lock:
                if has_evolved:
                    self._manifest.last_evolve_at = datetime.now(timezone.utc).isoformat()
                self._save_manifest(self._manifest)

    # ------------------------------------------------------------------ #
    #  Trigger Functions                                                 #
    # ------------------------------------------------------------------ #

    def _should_connect_merge(self) -> bool:
        """Check if the connect and merge phase should be triggered.
        
        Phase 1 (Connect and Merge) is triggered if the number of new clusters
        is greater than or equal to _EVO_CONNECT_MERGE_COUNT.
        """
        return len(self._manifest.connect_merge_cluster_ids) >= _EVO_CONNECT_MERGE_COUNT

    def _should_refresh_edges(self) -> bool:
        """Check if the edge refresh phase should be triggered.
        
        Phase 2 (Refresh Edges) is triggered if the number of clusters with
        refreshed embeddings is greater than or equal to _EVO_EDGE_REFRESH_COUNT.
        """
        return len(self._manifest.edge_refresh_cluster_ids) >= _EVO_EDGE_REFRESH_COUNT

    def _should_detect_meta_clusters(self) -> bool:
        """Check if the meta cluster detection phase should be triggered.
        
        Phase 3 (Detect Meta Clusters) is triggered if the number of steps
        since the last meta cluster detection is greater than or equal to
        _EVO_STEP_INTERVAL and the change in non-meta cluster count is greater
        than or equal to _EVO_META_DETECTION_COUNT.
        """
        # Check the step interval condition
        step_interval = self._manifest.current_step - self._manifest.last_meta_detection_step
        if step_interval < _EVO_STEP_INTERVAL:
            return False

        # Check the cluster count condition
        non_meta_cluster_count = self._get_non_meta_cluster_count()
        cluster_count_diff = abs(non_meta_cluster_count - self._manifest.last_meta_detection_cluster_count)
        return cluster_count_diff >= _EVO_META_DETECTION_COUNT

    def _should_global_update(self) -> bool:
        """Check if the global update phase should be triggered.
        
        Phase 4 (Global Update) is triggered if the number of steps since the
        last global update is greater than or equal to _EVO_STEP_INTERVAL and
        the change in non-meta cluster count is greater than or equal to
        _EVO_GLOBAL_UPDATE_COUNT.
        """
        # Check the step interval condition
        step_interval = self._manifest.current_step - self._manifest.last_global_update_step
        if step_interval < _EVO_STEP_INTERVAL:
            return False

        # Check the cluster count condition
        non_meta_cluster_count = self._get_non_meta_cluster_count()
        cluster_count_diff = abs(non_meta_cluster_count - self._manifest.last_global_update_cluster_count)
        return cluster_count_diff >= _EVO_GLOBAL_UPDATE_COUNT

    # ------------------------------------------------------------------ #
    #  Phase 1: Connect and Merge                                        #
    # ------------------------------------------------------------------ #

    @_managed_failure
    async def _connect_and_merge(self):
        """Phase 1: Connect and Merge for newly created clusters.

        For each newly created cluster, find other clusters with similar embeddings.
        If the similarity exceeds _EVO_CONNECT_SIMILARITY_THRESHOLD, create a weak
        semantic edge between them. If the similarity exceeds _EVO_MERGE_SIMILARITY_THRESHOLD,
        merge the clusters into a single cluster. Update the manifest with the new
        cluster IDs and clear the list of newly created clusters.
        """
        # Since these clusters are already in the storage, we need to pick them out, 
        # and perform connect & merge during putting them back in
        for cluster_id in self._manifest.connect_merge_cluster_ids:
            cluster, cluster_embedding = await self._storage.get_with_embedding(cluster_id)
            if cluster is None or cluster_embedding is None:
                continue

            # Find similar clusters based on embedding similarity
            # To avoid best match being in the list of newly created clusters, 
            # we search for top_k = len(connect_merge_cluster_ids) + 3
            best_match = None
            similar = await self._storage.search_similar_clusters(
                query_embedding=cluster_embedding,
                top_k=len(self._manifest.connect_merge_cluster_ids) + 3,
                similarity_threshold=_EVO_CONNECT_SIMILARITY_THRESHOLD
            )
            # Similar is sorted by similarity, so the first one is the best match
            if similar:
                filtered_similar = [
                    s for s in similar
                    if s["id"] not in self._manifest.connect_merge_cluster_ids
                ]
                if filtered_similar:
                    best_match = filtered_similar[0]
            
            # This cluster is not similar enough to connect or merge
            if best_match is None:
                continue

            # If a best match is found, discuss whether to connect or merge
            if best_match["similarity"] >= _EVO_MERGE_SIMILARITY_THRESHOLD:
                # Merge clusters
                best_cluster = await self._storage.get(best_match["id"])
                if best_cluster:
                    # Merge and update embedding
                    await self._merge_clusters(source=cluster, target=best_cluster)
            elif best_match["similarity"] >= _EVO_CONNECT_SIMILARITY_THRESHOLD:
                # Connect clusters with a weak semantic edge
                best_cluster = await self._storage.get(best_match["id"])
                if best_cluster:
                    # Create a weak semantic edge between the two clusters
                    await self._create_weak_semantic_edge(
                        source=cluster,
                        target=best_cluster,
                        edge_source="embed_sim",
                        weight=best_match["similarity"],
                    )
            else:
                # No action needed if similarity is below the threshold
                continue

        await self._log.info(
            f"[Evolve] Connected and merged clusters for {len(self._manifest.connect_merge_cluster_ids)} newly created clusters."
        )
        # Clear the list of newly created clusters after processing
        with self._manifest_guard():
            self._manifest.connect_merge_cluster_ids.clear()

    # ------------------------------------------------------------------ #
    #  Phase 2: Refresh Edges                                            #
    # ------------------------------------------------------------------ #

    @_managed_failure
    async def _refresh_edges(self):
        """Phase 2: Refresh Edges for clusters with updated embeddings.

        For each cluster with a refreshed embedding, find other clusters with similar embeddings.
        If the similarity exceeds _EVO_CONNECT_SIMILARITY_THRESHOLD, 
        create a weak semantic edge between them.
        """
        for cluster_id in self._manifest.edge_refresh_cluster_ids:
            cluster, cluster_embedding = await self._storage.get_with_embedding(cluster_id)
            if cluster is None or cluster_embedding is None:
                continue

            # Find similar clusters based on embedding similarity
            similar = await self._storage.search_similar_clusters(
                query_embedding=cluster_embedding,
                top_k=_EVO_EDGE_REFRESH_TOPK,
                similarity_threshold=_EVO_CONNECT_SIMILARITY_THRESHOLD
            )
            for similar_cluster in similar:
                if similar_cluster["id"] == cluster_id:
                    continue  # Skip self
                target_cluster = await self._storage.get(similar_cluster["id"])
                if target_cluster:
                    # Force update the weak semantic edge with the new similarity weight
                    await self._create_weak_semantic_edge(
                        source=cluster,
                        target=target_cluster,
                        edge_source="embed_sim",
                        weight=similar_cluster["similarity"],
                    )

        await self._log.info(
            f"[Evolve] Refreshed edges for {len(self._manifest.edge_refresh_cluster_ids)} clusters."
        )
        # Clear the list of clusters with refreshed embeddings after processing
        with self._manifest_guard():
            self._manifest.edge_refresh_cluster_ids.clear()

    # ------------------------------------------------------------------ #
    #  Phase 3: Detect Meta Clusters                                     #
    # ------------------------------------------------------------------ #

    @_managed_failure
    async def _detect_meta_clusters(self):
        """Phase 3: Detect Meta Clusters using Leiden community detection algorithm.

        Through the Leiden algorithm, the knowledge graph is partitioned into communities.
        For each community, if the number of clusters exceeds a threshold, a meta cluster
        is created to represent the community to reduce complexity during search.       
        """
        if ig is None or leidenalg is None:
            await self._log.warning(
                "[Evolver] igraph/leidenalg not installed; skipping meta-cluster detection"
            )
            return
        # Before running the meta cluster detection, we need to clean up existing meta clusters
        self._cleanup_meta_clusters()
        non_meta_cluster_count = self._get_non_meta_cluster_count()

        # Build edges for leiden algorithm
        raw_edges, weights = self._get_edges_and_weights()

        # Map cluster IDs to numeric indices for igraph
        # Single cluster with no edges will not be included in the community detection
        cluster_id_to_num, cluster_num_to_id, num = {}, {}, 0
        for source, target in raw_edges:
            if source not in cluster_id_to_num:
                cluster_id_to_num[source] = num
                cluster_num_to_id[num] = source
                num += 1
            if target not in cluster_id_to_num:
                cluster_id_to_num[target] = num
                cluster_num_to_id[num] = target
                num += 1

        # Run Leiden algorithm to detect communities
        edges = [(cluster_id_to_num[src], cluster_id_to_num[tgt]) for src, tgt in raw_edges]
        graph = ig.Graph(n=num, edges=edges, directed=False)
        partition = leidenalg.find_partition(
            graph=graph,
            partition_type=leidenalg.ModularityVertexPartition,
            weights=weights,
        )
        membership = partition.membership

        # Group clusters by community
        community_to_cluster_ids: Dict[int, List[str]] = defaultdict(list)
        for cluster_num, community_id in enumerate(membership):
            cluster_id = cluster_num_to_id.get(cluster_num, None)
            # If cluster_id is None, it means this cluster has no edges and
            # should not be included in any community. We skip it.
            if cluster_id is None:
                continue
            community_to_cluster_ids[community_id].append(cluster_id)
    
        # Build meta clusters for each community with more than 1 cluster
        meta_cluster_coroutines = []
        for community_id, cluster_ids in community_to_cluster_ids.items():
            if len(cluster_ids) <= 1:
                continue

            meta_cluster_coroutines.append(self._build_meta_cluster(community_id, cluster_ids))
        
        await self._gather_phase(meta_cluster_coroutines)

        await self._log.info(
            f"[Evolve] Created {len(community_to_cluster_ids)} meta clusters "
            f"from {non_meta_cluster_count} non-meta clusters."
        )

        # Update the last meta detection step and cluster count in the manifest
        with self._manifest_guard():
            self._manifest.last_meta_detection_step = self._manifest.current_step
            self._manifest.last_meta_detection_cluster_count = non_meta_cluster_count

    # ------------------------------------------------------------------ #
    #  Phase 4: Global Update                                            #
    # ------------------------------------------------------------------ #

    @_managed_failure
    async def _global_update(self):
        """Phase 4: Global Update to ensure the knowledge graph is up-to-date.

        Since Phase 2 does not remove existing edges with low similarity,
        for each edge (source = embed_sim), we need to re-calculate the similarity.
        If the similarity falls below _EVO_CONNECT_SIMILARITY_THRESHOLD, remove the edge.
        
        Then, refine the clusters with hotness and confidence below the thresholds.
        There are at most _EVO_GLOBAL_UPDATE_MAX_CLUSTERS clusters to consider for global update in a single step.
        
        Finally, perform Phase 1, 2 and 3 again to ensure the knowledge graph is up-to-date.
        """
        # Update the edges and remove the invalid edges based on the similarity threshold
        embed_sim_edges = self._get_all_embed_sim_edges()

        updates = defaultdict(dict)  # source_id -> {target_id: new_weight}
        for (source_id, target_id, old_weight, new_weight) in embed_sim_edges:
            if old_weight == new_weight:
                continue  # No change in weight, skip
            updates[source_id][target_id] = new_weight
            updates[target_id][source_id] = new_weight

        # Update the edges in the storage
        for source_id, edge_updates in updates.items():
            cluster = await self._storage.get(source_id)
            if cluster is None:
                continue
            # Update the edges with new weights and remove edges below the threshold
            updated_related_clusters = []
            for edge in cluster.related_clusters:
                if edge.source == "embed_sim" and edge.target_cluster_id in edge_updates:
                    if edge_updates[edge.target_cluster_id] >= _EVO_GLOBAL_UPDATE_DISCONNECT_SIMILARITY_THRESHOLD:
                        edge.weight = edge_updates[edge.target_cluster_id]
                        updated_related_clusters.append(edge)
                else:
                    updated_related_clusters.append(edge)
            cluster.related_clusters = updated_related_clusters
            self._require_write(await self._storage.update(cluster))

        # Refine the clusters with hotness and confidence below the thresholds
        cluster_ids_to_refine = self._get_cluster_ids_to_refine()
        refine_coroutines = [
            self._refine_cluster(cluster_id) for cluster_id in cluster_ids_to_refine
        ]
        await self._gather_phase(refine_coroutines)

        # Perform phase 1, 2 and 3 again to ensure the knowledge graph is up-to-date
        await self._connect_and_merge()
        await self._refresh_edges()
        await self._detect_meta_clusters()

        # Update the last global update step and cluster count in the manifest
        non_meta_cluster_count = self._get_non_meta_cluster_count()
        with self._manifest_guard():
            self._manifest.last_global_update_step = self._manifest.current_step
            self._manifest.last_global_update_cluster_count = non_meta_cluster_count

    # ------------------------------------------------------------------ #
    #  Shared Helper Functions                                           #
    # ------------------------------------------------------------------ #

    def _manifest_guard(self):
        return self._runtime.io_lock if self._runtime is not None else nullcontext()

    def _ensure_writable(self):
        if self._runtime is not None:
            if self._runtime.failure:
                raise LifecycleError(self._runtime.failure)
            if self._runtime.closing or self._runtime.closed or self._runtime.frozen:
                raise LifecycleError("managed_evolution_not_writable")

    def _require_write(self, result):
        if self._runtime is not None and result is False:
            raise LifecycleError("evolution_storage_write_failed")

    @_managed_failure
    async def _gather_phase(self, coroutines):
        # Settle *all* siblings before raising, so the owning step/lease cannot
        # finish while another coroutine is still applying storage changes.
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        if self._runtime is not None:
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        return results

    def _get_non_meta_cluster_count(self) -> int:
        """Get the count of non-meta clusters in the knowledge storage."""
        result = self._storage.db.fetch_one(
            f"SELECT COUNT(*) FROM {self._storage.table_name} "
            f"WHERE lifecycle != ?", [Lifecycle.META.name]
        )
        return result[0] if result else 0

    @_managed_failure
    async def _update_cluster_embedding(self, cluster: KnowledgeCluster):
        """Update the embedding of a cluster based on its queries."""
        if not cluster.queries:
            return
        try:
            combined_text = self._storage.combine_cluster_fields(cluster.queries)
            text_hash = compute_text_hash(combined_text)
            embedding_vector = (await self._embedding.embed([combined_text]))[0]

            self._require_write(await self._storage.store_embedding(
                cluster_id=cluster.id,
                embedding_vector=embedding_vector,
                embedding_model=self._embedding.model_id,
                embedding_text_hash=text_hash
            ))
        except Exception as e:
            if self._runtime is not None:
                raise
            await self._log.error(f"Failed to update embedding for cluster {cluster.id}: {e}")
            return

    # ------------------------------------------------------------------ #
    #  Phase 1 Helper Functions                                          #
    # ------------------------------------------------------------------ #

    @_managed_failure
    async def _merge_clusters(self, source: KnowledgeCluster, target: KnowledgeCluster):
        """Merge source cluster into target cluster and update the storage."""
        merged_cluster = await self._storage.merge([target, source])
        if merged_cluster is None:
            if self._runtime is not None:
                raise LifecycleError("evolution_storage_merge_failed")
            return
        # Update embedding with merged queries
        await self._update_cluster_embedding(merged_cluster)
        with self._manifest_guard():
            self._manifest.edge_refresh_cluster_ids.append(merged_cluster.id)

        # Update lifecycle if merge count exceeds threshold
        if merged_cluster.merge_count >= 3 and merged_cluster.lifecycle == Lifecycle.EMERGING:
            merged_cluster.lifecycle = Lifecycle.STABLE
        self._require_write(await self._storage.update(merged_cluster))

    # ------------------------------------------------------------------ #
    #  Phase 2 Helper Functions                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _add_edge(
        cluster: KnowledgeCluster,
        target_cluster_id: str, 
        edge_source: str,
        weight: float,
    ):
        """Add a weak semantic edge to the cluster or update the weight"""
        for edge in cluster.related_clusters:
            if edge.target_cluster_id == target_cluster_id and edge.source == edge_source:
                edge.weight = max(edge.weight, weight)
                return
        cluster.related_clusters.append(
            WeakSemanticEdge(target_cluster_id=target_cluster_id, weight=weight, source=edge_source)
        )

    @_managed_failure
    async def _create_weak_semantic_edge(
        self,
        source: KnowledgeCluster,
        target: KnowledgeCluster,
        edge_source: str,
        weight: float,
    ):
        """Create a weak semantic edge between two clusters and update the storage."""
        self._add_edge(source, target.id, edge_source, weight)
        self._add_edge(target, source.id, edge_source, weight)
        # Update the clusters in storage
        self._require_write(await self._storage.update(source))
        self._require_write(await self._storage.update(target))

    # ------------------------------------------------------------------ #
    #  Phase 3 Helper Functions                                          #
    # ------------------------------------------------------------------ #

    def _cleanup_meta_clusters(self):
        """Remove all existing meta clusters from the knowledge storage."""
        # Since we don't build edge from other clusters to meta clusters 
        # to avoid breaking change in one-hop expansion,
        # we do not need to remove edges pointing to meta clusters.

        # remove meta clusters themselves via DuckDB query to avoid N+1 queries
        self._storage.db.execute(
            f"DELETE FROM {self._storage.table_name} "
            f"WHERE lifecycle = ?", [Lifecycle.META.name]
        )

    def _get_edges_and_weights(self) -> Tuple[List[Tuple[str, str]], List[float]]:
        """Get all edges and corresponding weights via DuckDB query to avoid N+1 queries."""
        query = f"""
        SELECT kc.id                                         AS source,
            json_extract_string(edge, '$.target_cluster_id') AS target,
            CAST(json_extract(edge, '$.weight') AS DOUBLE)   AS weight
        FROM {self._storage.table_name} kc,
            unnest(CAST(kc.related_clusters::JSON AS JSON[])) AS t(edge)
        WHERE kc.related_clusters IS NOT NULL
        AND kc.related_clusters != '[]'
        AND kc.related_clusters != ''
        AND kc.id < json_extract_string(edge, '$.target_cluster_id')
        """
        rows = self._storage.db.fetch_all(query)
        edges, weights = [], []
        for row in rows:
            edges.append((row[0], row[1]))
            weights.append(row[2])
        return edges, weights

    @_managed_failure
    async def _build_meta_cluster(self, community_id: int, cluster_ids: List[str]):
        """Build a meta cluster for a given community of clusters."""
        async with self._llm_semaphore:
            meta_cluster_id = f"M{community_id:04d}"
            # Fetch queries and abstraction_level from the clusters in the community
            if _EVO_META_DETECTION_MAX_QUERIES >= len(cluster_ids):
                num_queries_per_cluster = _EVO_META_DETECTION_MAX_QUERIES // len(cluster_ids)
            else:
                num_queries_per_cluster = 1

            rows = self._storage.db.fetch_one(f"""
                WITH matched AS (
                    SELECT queries, abstraction_level FROM {self._storage.table_name}
                    WHERE id IN (SELECT unnest(?::VARCHAR[]))
                ),
                limited_queries AS (
                    SELECT t.queries
                    FROM matched, unnest(CAST(queries::JSON AS JSON[])) WITH ORDINALITY AS t(queries, row_num)
                    WHERE row_num <= ?
                )
                SELECT
                    ( SELECT json_group_array(queries) FROM limited_queries ) AS cluster_queries,
                    ( SELECT abstraction_level
                    FROM matched
                    WHERE abstraction_level IS NOT NULL
                    GROUP BY abstraction_level
                    ORDER BY COUNT(*) DESC
                    LIMIT 1
                    ) AS most_common_abstraction_level
            """, [cluster_ids, num_queries_per_cluster])
            if rows:
                cluster_queries, most_common_abstraction_level = rows[0], rows[1]
            else:
                if self._runtime is not None:
                    raise LifecycleError("evolution_meta_queries_missing")
                await self._log.error(f"[Evolve] Failed to fetch queries for meta cluster {meta_cluster_id}.")
                return

            # Build query for the meta cluster by summary common queries from the community clusters
            queries = json.loads(cluster_queries) if cluster_queries else []
            queries = queries[:_EVO_META_DETECTION_MAX_QUERIES]  # Limit the number of queries for the prompt
            prompt = EVO_META_QUERY.format(
                queries="\n".join(queries)
            )
            response = await self._llm.achat([{"role": "user", "content": prompt}])
            meta_query = response.content.strip()

            # Build related clusters
            related_clusters = [
                WeakSemanticEdge(target_cluster_id=cluster_id, weight=1.0, source="meta")
                for cluster_id in cluster_ids
            ]

            # Build the meta cluster object
            # For meta clusters, the queries and embeddings are the most important
            if most_common_abstraction_level is not None:
                most_common_abstraction_level = AbstractionLevel[most_common_abstraction_level]

            meta_cluster = KnowledgeCluster(
                id=meta_cluster_id,
                name=meta_query,
                description=[f"Meta cluster ({len(cluster_ids)} clusters) for: {meta_query}"],
                content="Meta cluster",
                confidence=0.5,
                abstraction_level=most_common_abstraction_level,
                hotness=0.5,
                lifecycle=Lifecycle.META,
                related_clusters=related_clusters,
                queries=[meta_query],
            )

            self._require_write(await self._storage.insert(meta_cluster))
            await self._update_cluster_embedding(meta_cluster)

    # ------------------------------------------------------------------ #
    #  Phase 4 Helper Functions                                          #
    # ------------------------------------------------------------------ #

    def _get_all_embed_sim_edges(self) -> List[Tuple[str, str, float, float]]:
        """Get all edges with their old and new weights for global update phase."""
        dim = self._storage.embedding_dim
        query = f"""
        SELECT
            kc.id AS source_id,
            json_extract_string(edge, '$.target_cluster_id') AS target_id,
            CAST(json_extract(edge, '$.weight') AS DOUBLE) AS old_weight,
            list_cosine_similarity(
                kc.embedding_vector::FLOAT[{dim}],
                target.embedding_vector::FLOAT[{dim}]
            ) AS new_weight
        FROM {self._storage.table_name} kc,
             unnest(CAST(kc.related_clusters::JSON AS JSON[])) AS t(edge),
             {self._storage.table_name} target
        WHERE json_extract_string(edge, '$.source') = 'embed_sim'
          AND target.id = json_extract_string(edge, '$.target_cluster_id')
          AND target.embedding_vector IS NOT NULL
          AND kc.embedding_vector IS NOT NULL
          AND kc.id < target.id
        """
        return self._storage.db.fetch_all(query)

    def _get_cluster_ids_to_refine(self) -> List[str]:
        """Get cluster IDs to be refined based on hotness and confidence thresholds."""
        query = f"""
        SELECT id FROM {self._storage.table_name}
        WHERE lifecycle != ?
          AND hotness < ?
          AND confidence < ?
        ORDER BY hotness + confidence ASC
        LIMIT ?
        """
        parameters = [
            Lifecycle.META.name,
            _EVO_GLOBAL_UPDATE_HOTNESS_THRESHOLD,
            _EVO_GLOBAL_UPDATE_CONFIDENCE_THRESHOLD,
            _EVO_GLOBAL_UPDATE_MAX_CLUSTERS
        ]
        rows = self._storage.db.fetch_all(query, parameters)
        return [row[0] for row in rows]

    @_managed_failure
    async def _refine_cluster(self, cluster_id: str):
        """Refine a cluster by invoking the LLM to improve its queries and content."""
        async with self._llm_semaphore:
            cluster = await self._storage.get(cluster_id)
            if cluster is None:
                await self._log.warning(f"[Evolve] Cluster {cluster_id} not found for refinement.")
                return

            # Build prompt for the LLM to refine the cluster's queries and content
            prompt = EVO_REFINE_CLUSTER.format(
                queries=self._storage.combine_cluster_fields(cluster.queries),
                content=str(cluster.content)[:3000]
            )

            try:
                response = await self._llm.achat([{"role": "user", "content": prompt}])
                # Parse the response
                raw_content = response.content.strip() or ""
                refined_data = self._extract_json_object(raw_content)
                query = refined_data["query"]
                content = refined_data["content"]

                # Update the cluster with refined data
                cluster.queries = [query]
                cluster.content = content
                cluster.hotness = _EVO_GLOBAL_UPDATE_REFINED_HOTNESS
                cluster.confidence = _EVO_GLOBAL_UPDATE_REFINED_CONFIDENCE
                cluster.lifecycle = Lifecycle.EMERGING

                # Update the storage and embedding
                self._require_write(await self._storage.update(cluster))
                await self._update_cluster_embedding(cluster)
                with self._manifest_guard():
                    self._manifest.edge_refresh_cluster_ids.append(cluster_id)

            except Exception as e:
                if self._runtime is not None:
                    raise
                await self._log.error(f"[Evolve] Failed to parse LLM response for cluster {cluster_id}: {e}")
                return

    def _extract_json_object(self, text: str) -> Optional[Dict]:
        """Extract the first JSON object from a string."""
        start = text.index('{')
        end = text.rindex('}')
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    # ------------------------------------------------------------------ #
    #  Manifest I/O                                                      #
    # ------------------------------------------------------------------ #
    def _load_manifest(self) -> EvolveManifest:
        if self._runtime is not None:
            with self._runtime.io_lock:
                try:
                    content = self._manifest_path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    return EvolveManifest()
                except BaseException:
                    self._runtime.fail("evolve_manifest_read_failed")
                    raise
                try:
                    return EvolveManifest.from_json(content, strict=True)
                except BaseException:
                    self._runtime.fail("evolve_manifest_invalid")
                    raise
        if self._manifest_path.exists():
            try:
                return EvolveManifest.from_json(
                    self._manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass
        return EvolveManifest()

    def _save_manifest(self, manifest: EvolveManifest) -> None:
        """Atomically persist the manifest via write-to-tmp + rename.

        This prevents partial JSON on disk if the process is killed mid-write.
        """
        if self._runtime is not None:
            with self._runtime.io_lock:
                self._ensure_writable()
                try:
                    # Validate memory too; do not publish malformed state that
                    # the next supervised startup would have to reject.
                    EvolveManifest.from_json(manifest.to_json(), strict=True)
                    if manifest is not self._runtime._evolve_manifest:
                        raise LifecycleError("evolve_manifest_not_authoritative")
                    write_json_atomic(self._manifest_path, asdict(manifest))
                except BaseException:
                    self._runtime.fail("evolve_manifest_write_failed")
                    raise
            return
        tmp_path = self._manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(manifest.to_json(), encoding="utf-8")
        tmp_path.replace(self._manifest_path)

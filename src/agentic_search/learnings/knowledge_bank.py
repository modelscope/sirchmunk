import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.agentic_search.learnings.evidence_processor import (
    MonteCarloEvidenceSampling,
    RoiResult,
)
from src.agentic_search.llm.openai import OpenAIChat
from src.agentic_search.llm.prompts import EVIDENCE_SUMMARY
from src.agentic_search.schema.knowledge import (
    AbstractionLevel,
    EvidenceUnit,
    KnowledgeCluster,
    Lifecycle,
)
from src.agentic_search.schema.metadata import FileInfo
from src.agentic_search.schema.request import Request
from src.agentic_search.storage.knowledge_storage import KnowledgeStorage
from src.agentic_search.utils.file_utils import StorageStructure, fast_extract
from src.agentic_search.utils.utils import extract_fields

# In-memory knowledge storage, keyed by cluster ID
_KNOWLEDGE_MAP: Dict[str, KnowledgeCluster] = {}

# _LATEST_KNOWLEDGE_NUMERIC_IDX: int = 0


class KnowledgeBank:
    """
    A knowledge bank that manages knowledge clusters built from retrieved information and metadata dynamically.
    """

    def __init__(
        self,
        llm: OpenAIChat,
        metadata_map: Dict[str, Any] = None,
        work_path: Union[str, Path] = None,
    ):
        """
        Initialize the KnowledgeBank with an LLM and metadata mapping.

        Args:
            llm (OpenAIChat): An instance of the OpenAIChat LLM for processing text.
            metadata_map (Dict[str, Any]): A mapping of all metadata information.
                k: metadata cache key, refers to `FileInfo.cache_key`
                v: metadata path or content
        """
        self.llm = llm
        self.metadata_map = metadata_map
        self.work_path: Path = (
            Path.cwd() if work_path is None else Path(work_path).resolve()
        )
        self.metadata_path: Path = (
            self.work_path / StorageStructure.CACHE_DIR / StorageStructure.METADATA_DIR
        )

        self.knowledge_storage = KnowledgeStorage(
            work_path=self.work_path, readonly=False
        )

    @staticmethod
    def _get_file_info(
        file_or_url: str, metadata_path: Union[str, Path]
    ) -> Optional[FileInfo]:

        cache_key: str = FileInfo.get_cache_key(file_or_url=file_or_url)
        meta_file: Path = Path(metadata_path) / f"{cache_key}.json"

        if not meta_file.exists():
            return None

        with open(meta_file, "r", encoding="utf-8") as f:
            metadata_content = json.load(f)

        return FileInfo.from_dict(info=metadata_content)

    async def build(
        self,
        request: Request,
        retrieved_infos: List[Dict[str, Any]],
        keywords: Dict[str, float] = None,
        top_k_files: Optional[int] = 3,
        top_k_snippets: Optional[int] = 5,
        confidence_threshold: Optional[float] = 8.0,
        verbose: bool = True,
    ) -> Union[KnowledgeCluster, None]:
        """Build a knowledge cluster from retrieved information and metadata dynamically."""

        if len(retrieved_infos) == 0:
            logger.warning(
                "No retrieved information available to build knowledge cluster."
            )
            return None

        retrieved_infos = retrieved_infos[:top_k_files]

        keywords = keywords or {}

        # Get evidence units (regions of interest) from raw retrieved infos
        evidences: List[EvidenceUnit] = []
        for info in retrieved_infos:
            file_path_or_url: str = info["path"]

            # TODO: handle more file types; deal with large files; Async adaptive
            extraction_result = await fast_extract(file_path=file_path_or_url)
            doc_content: str = extraction_result.content

            sampler = MonteCarloEvidenceSampling(
                llm=self.llm,
                doc_content=doc_content,
                verbose=verbose,
            )
            roi_result: RoiResult = await sampler.get_roi(
                query=request.get_user_input(),
                keywords=keywords,
                confidence_threshold=confidence_threshold,
                top_k=top_k_snippets,
            )

            evidence_unit = EvidenceUnit(
                doc_id=FileInfo.get_cache_key(file_path_or_url),
                file_or_url=Path(file_path_or_url),
                summary=roi_result.summary,
                is_found=roi_result.is_found,
                snippets=roi_result.snippets,
                extracted_at=datetime.now(),
                conflict_group=[],
            )
            evidences.append(evidence_unit)

        if len(evidences) == 0:
            logger.warning("No evidence units extracted from retrieved information.")
            return None

        # Get `name`, `description` and `content` from user request and evidences using LLM
        # TODO: to be processed other type of segments
        evidence_contents: List[str] = [ev.summary for ev in evidences]

        evidence_summary_prompt: str = EVIDENCE_SUMMARY.format(
            user_input=request.get_user_input(),
            evidences="\n\n".join(evidence_contents),
        )

        evidence_summary_response: str = await self.llm.achat(
            messages=[{"role": "user", "content": evidence_summary_prompt}],
            stream=True,
        )

        cluster_infos: Dict[str, Any] = extract_fields(
            content=evidence_summary_response
        )
        if len(cluster_infos) == 0:
            logger.warning(
                "Failed to extract knowledge cluster information from LLM response."
            )
            return None

        cluster = KnowledgeCluster(
            id=f"C{len(_KNOWLEDGE_MAP) + 1}",
            name=cluster_infos.get("name"),
            description=[cluster_infos.get("description")],
            content=cluster_infos.get("content"),
            scripts=[],
            resources=[],
            patterns=[],
            constraints=[],
            evidences=evidences,
            confidence=0.5,
            abstraction_level=AbstractionLevel.TECHNIQUE,
            landmark_potential=0.5,
            hotness=0.5,
            lifecycle=Lifecycle.EMERGING,
            create_time=datetime.now(),
            last_modified=datetime.now(),
            version=1,
            related_clusters=[],
        )

        return cluster

    async def get(self, cluster_id: str) -> Optional[KnowledgeCluster]:
        """
        Get knowledge cluster by ID.
        """
        return _KNOWLEDGE_MAP.get(cluster_id, None)

    async def update(self, cluster: KnowledgeCluster):
        """
        Update or add a knowledge cluster in the in-memory storage.
        """

        if cluster is not None:
            _KNOWLEDGE_MAP[cluster.id] = cluster

    async def save(self, cluster: KnowledgeCluster):
        """
        Save a knowledge cluster to persistent storage.
        """
        self.knowledge_storage.insert_cluster(cluster=cluster)

    async def merge(self, clusters: List[KnowledgeCluster]) -> KnowledgeCluster:
        """
        Merge multiple similar knowledge clusters into a single cluster.
        """
        ...

    async def split(self, cluster: KnowledgeCluster) -> List[KnowledgeCluster]:
        """
        Split a knowledge cluster into more focused clusters.
        """
        ...

    async def remove(self, cluster_id: str) -> None:
        """
        Remove a knowledge cluster by ID from the in-memory storage.
        """
        if cluster_id in _KNOWLEDGE_MAP:
            del _KNOWLEDGE_MAP[cluster_id]
        else:
            logger.warning(
                "Knowledge cluster with ID {} not found for removal.", cluster_id
            )

    async def clear(self) -> None:
        """
        Clear all knowledge clusters from the in-memory storage.
        """
        _KNOWLEDGE_MAP.clear()

    async def find(self, query: str) -> List[KnowledgeCluster]:
        """
        Find knowledge clusters relevant to the query.  TODO ...
        """
        ...

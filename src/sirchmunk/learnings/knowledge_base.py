# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union


from sirchmunk.learnings.evidence_processor import (
    MonteCarloEvidenceSampling,
    RoiResult,
)
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import EVIDENCE_SUMMARY
from sirchmunk.schema.knowledge import (
    AbstractionLevel,
    EvidenceUnit,
    KnowledgeCluster,
    Lifecycle,
)
from sirchmunk.schema.metadata import FileInfo
from sirchmunk.schema.request import Request
from sirchmunk.utils.constants import DEFAULT_SIRCHMUNK_WORK_PATH
from sirchmunk.utils.file_utils import StorageStructure, fast_extract
from sirchmunk.utils import create_logger, LogCallback
from sirchmunk.utils.utils import extract_fields

class KnowledgeBase:
    """
    A knowledge base that manages knowledge clusters built from retrieved information and metadata dynamically.
    """

    def __init__(
        self,
        llm: OpenAIChat,
        metadata_map: Dict[str, Any] = None,
        work_path: Union[str, Path] = None,
        log_callback: LogCallback = None,
    ):
        """
        Initialize the KnowledgeBase with an LLM and metadata mapping.

        Args:
            llm (OpenAIChat): An instance of the OpenAIChat LLM for processing text.
            metadata_map (Dict[str, Any]): A mapping of all metadata information.
                k: metadata cache key, refers to `FileInfo.cache_key`
                v: metadata path or content
            work_path: Working directory path
            log_callback: Optional log callback function for custom logging
        """
        self.llm = llm
        self.metadata_map = metadata_map
        self.work_path: Path = (
            DEFAULT_SIRCHMUNK_WORK_PATH if work_path is None else Path(work_path).resolve()
        )
        self.metadata_path: Path = (
            self.work_path / StorageStructure.CACHE_DIR / StorageStructure.METADATA_DIR
        )
        
        # Store log_callback for passing to child components
        self.log_callback = log_callback
        
        # Create bound logger with callback - returns AsyncLogger instance
        self._log = create_logger(log_callback=log_callback)

        self.llm_usages: List[Dict[str, Any]] = []

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

    @staticmethod
    def _compose_cluster_text(
        name: Optional[str],
        description: Union[List[str], str, None],
        content: Union[List[str], str, None],
    ) -> str:
        """
        Compose a stable text representation of a cluster from name, description, and content.
        This is used for deterministic cluster ID generation.
        """
        parts: List[str] = []
        if name:
            parts.append(str(name).strip())

        if description:
            if isinstance(description, list):
                parts.extend([str(item).strip() for item in description if item])
            else:
                parts.append(str(description).strip())

        if content:
            if isinstance(content, list):
                parts.extend([str(item).strip() for item in content if item])
            else:
                parts.append(str(content).strip())

        return "\n\n".join([part for part in parts if part])

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
            await self._log.warning(
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
                log_callback=self.log_callback,
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
            self.llm_usages.extend(sampler.llm_usages)
            evidences.append(evidence_unit)

        if len(evidences) == 0:
            await self._log.warning("No evidence units extracted from retrieved information.")
            return None

        # Get `name`, `description` and `content` from user request and evidences using LLM
        # TODO: to be processed other type of segments
        evidence_contents: List[str] = [ev.summary for ev in evidences]

        evidence_summary_prompt: str = EVIDENCE_SUMMARY.format(
            user_input=request.get_user_input(),
            evidences="\n\n".join(evidence_contents),
        )

        evidence_summary_llm_response = await self.llm.achat(
            messages=[{"role": "user", "content": evidence_summary_prompt}],
            stream=True,
        )
        evidence_summary_response: str = evidence_summary_llm_response.content
        self.llm_usages.append(evidence_summary_llm_response.usage)

        cluster_infos: Dict[str, Any] = extract_fields(
            content=evidence_summary_response
        )
        if len(cluster_infos) == 0:
            await self._log.warning(
                "Failed to extract knowledge cluster information from LLM response."
            )
            return None

        cluster_name = cluster_infos.get("name")
        cluster_description = cluster_infos.get("description")
        cluster_content = cluster_infos.get("content")

        cluster_text = self._compose_cluster_text(
            name=cluster_name,
            description=cluster_description,
            content=cluster_content,
        )
        if not cluster_text:
            cluster_text = request.get_user_input() or "unknown"

        cluster_id = f"C{hashlib.sha256(cluster_text.encode('utf-8')).hexdigest()[:10]}"

        # TODO: Adapt cluster attributes based on real scenarios
        cluster = KnowledgeCluster(
            id=cluster_id,
            name=cluster_name,
            description=[cluster_description] if cluster_description else [],
            content=cluster_content,
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

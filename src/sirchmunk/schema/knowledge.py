# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class Lifecycle(Enum):
    """Lifecycle status of the knowledge cluster, used by Cognition Layer for path planning."""

    STABLE = "stable"
    EMERGING = "emerging"
    CONTESTED = "contested"
    DEPRECATED = "deprecated"


class AbstractionLevel(Enum):
    """Abstraction tier for cognitive mapping and navigation depth control."""

    TECHNIQUE = 1  # e.g., QLoRA, Speculative Decoding
    PRINCIPLE = 2  # e.g., Low-Rank Update, Token Pruning
    PARADIGM = 3  # e.g., Parameter-Efficient Learning
    FOUNDATION = 4  # e.g., Gradient-Based Optimization
    PHILOSOPHY = 5  # e.g., Occam's Razor in ML


@dataclass
class EvidenceUnit:
    """
    Lightweight reference to an evidence unit in the Evidence Layer.

    Enables traceability and dynamic validation without storing full content.
    """

    # ID of the source doc, pointing to FileInfo cache key (FileInfo.get_cache_key())
    # If URL-based, this can be a hash of the URL
    doc_id: str

    # Path or URL to the source document
    # From `file_or_url` in FileInfo
    file_or_url: Union[str, Path]

    # Summarized of snippets
    summary: str

    # Segment within the document (e.g., paragraph, code snippet)
    is_found: bool

    # Segments within the document (e.g., paragraph, code snippet)
    # Format: {"snippet": "xxx", "start": 7, "end": 65, "score": 9.0, "reasoning": "xxx"}
    snippets: List[Dict[str, Any]]

    # When this evidence was processed
    extracted_at: datetime

    # IDs of conflict group if this evidence contradicts others
    conflict_group: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize EvidenceUnit to a dictionary.
        """
        return {
            "doc_id": self.doc_id,
            "file_or_url": str(self.file_or_url),
            "summary": self.summary,
            "is_found": self.is_found,
            "snippets": self.snippets,
            "extracted_at": self.extracted_at.isoformat(),
            "conflict_group": self.conflict_group,
        }


@dataclass
class Constraint:
    """
    Boundary condition for safe application of the cluster's conclusions.

    Used by Cognition Layer to activate/deactivate Barrier edges.
    """

    # DSL expression, e.g., "data_size < 100", "model_arch == 'decoder'"
    condition: str

    # "low", "medium", "high" — affects path blocking in Cognition Layer
    severity: str

    # Human-readable explanation of the constraint
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize Constraint to a dictionary.
        """
        return {
            "condition": self.condition,
            "severity": self.severity,
            "description": self.description,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Constraint":
        """
        Deserialize Constraint from a dictionary.
        """
        return Constraint(
            condition=data["condition"],
            severity=data["severity"],
            description=data["description"],
        )


@dataclass
class WeakSemanticEdge:
    """
    A lightweight, statistical, probabilistic association to another cluster. Undirected edge.

    Used for:
      - Fast nearest-neighbor search (e.g., "similar topics")
      - Cognitive map layout (force-directed positioning)
      - Fallback path suggestion when rich edges fail
      - Cold-start cluster grouping (e.g., for multi-cluster nodes)

    Weight semantics depend on source:
      - co_occur: P(B | A) from evidence co-mention
      - query_seq: P(next=B | current=A) from user logs
      - embed_sim: cosine similarity of cluster embeddings (if temporarily used)
    """

    target_cluster_id: str  # e.g., "C1005"
    weight: float  # [0.0, 1.0]; higher = stronger association
    source: str  # e.g., "co_occur", "query_seq", "embed_sim"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize WeakSemanticEdge to a dictionary.
        """
        return {
            "target_cluster_id": self.target_cluster_id,
            "weight": self.weight,
            "source": self.source,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WeakSemanticEdge":
        """
        Deserialize WeakSemanticEdge from a dictionary.
        """
        return WeakSemanticEdge(
            target_cluster_id=data["target_cluster_id"],
            weight=data["weight"],
            source=data["source"],
        )


@dataclass
class KnowledgeCluster:
    """
    A high-level, dynamic, consensus-based knowledge unit distilled from multiple evidence sources.

    Long-term memory notes with persistence.

    Serves as the bridge between raw evidence (Evidence Layer) and cognitive navigation (Cognition Layer).
    Designed for efficient retrieval, evolution, and integration into a cognitive map.
    """

    # Globally unique cluster ID, e.g., "C1007"
    id: str

    # Concise, human-readable name, e.g., "QLoRA: 4-bit Quantized Low-Rank Adaptation"
    name: str

    # Detailed abstract/summary of the knowledge cluster from different perspectives
    description: Union[str, List[str]]

    # The markdown main content of the knowledge cluster, could be table of contents, references, etc.
    content: Union[str, List[str]]

    # Optional code snippets to process or demonstrate the knowledge
    # Each item should be standard code string (e.g., Python, Bash) with function annotations.
    scripts: Optional[List[str]] = None

    # Related resources such as files, URLs
    # Example: [{"type": "url", "value": "https://example.com"}, {"type": "file", "value": "/path/to/file"}]
    resources: Optional[List[Dict[str, Any]]] = None

    # References to supporting evidence items (e.g., paragraphs, code snippets)
    evidences: List[EvidenceUnit] = field(default_factory=list)

    # 3–5 generalizable design principles or mechanisms
    patterns: List[str] = field(default_factory=list)

    # Boundary conditions for safe application
    constraints: List[Constraint] = field(default_factory=list)

    # Total consensus score: aggregated from evidence weights, co-occurrence, and consistency.
    # Range: [0.0, 1.0]; dynamically updated upon ingestion of new evidence.
    confidence: Optional[float] = None

    # Abstraction level (e.g., "conceptual", "architectural", "implementation");
    # guides hierarchy placement and path traversal depth in the cognitive map.
    abstraction_level: Optional[AbstractionLevel] = None

    # Estimated suitability as a cognitive landmark node (e.g., for navigation shortcuts).
    # Range: [0.0, 1.0]
    landmark_potential: Optional[float] = None

    # Activity score reflecting query coverage or update frequency.
    # Range: [0.0, 1.0]
    hotness: Optional[float] = None

    # Structural lifecycle classification of knowledge cluster node
    # critical for path planning and validity of cached shortcuts in Cognition Layer.
    lifecycle: Lifecycle = Lifecycle.EMERGING

    # ISO 8601 timestamp of creation
    create_time: Optional[datetime] = None

    # ISO 8601 timestamp of last structural or semantic update
    last_modified: Optional[datetime] = None

    # Version number; incremented on structural changes (e.g., pattern/constraint updates)
    version: Optional[int] = None

    # Related knowledge clusters for estimated weak semantic connections
    related_clusters: List[WeakSemanticEdge] = None

    def __post_init__(self):
        if self.related_clusters is None:
            self.related_clusters = []

        if self.create_time is None:
            self.create_time = datetime.now(timezone.utc)

        if self.last_modified is None:
            self.last_modified = datetime.now(timezone.utc)

        if self.version is None:
            self.version = 0

    @property
    def primary_evidence_files(self) -> Set[str]:
        """Return set of unique file IDs backing this cluster — useful for evidence-layer prefetch."""
        return {ref.doc_id for ref in self.evidences}

    def get_conflict_groups(self) -> Set[str]:
        """Extract conflict group IDs for cognitive conflict-aware reasoning."""
        return {ref.conflict_group for ref in self.evidences if ref.conflict_group}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize KnowledgeCluster to a dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "scripts": self.scripts,
            "resources": self.resources,
            "patterns": self.patterns,
            "constraints": [c.to_dict() for c in self.constraints],
            "evidences": [er.to_dict() for er in self.evidences],
            "confidence": self.confidence,
            "abstraction_level": self.abstraction_level.name,
            "landmark_potential": self.landmark_potential,
            "hotness": self.hotness,
            "lifecycle": self.lifecycle.name,
            "create_time": self.create_time.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "version": self.version,
            "related_clusters": [rc.to_dict() for rc in self.related_clusters],
        }


if __name__ == "__main__":

    # Create instance
    cluster = KnowledgeCluster(
        id="C1001",
        name="Test Cluster",
        description=["A desc from perspective A.", "A desc from perspective B."],
        content="Detailed content of the knowledge cluster.",
        scripts=["print('Hello World')"],
        resources=[
            {"type": "url", "value": "https://example.com"},
            {"type": "file", "value": "/data/image1.png"},
        ],
        patterns=["pattern A", "pattern B"],
        constraints=[Constraint("x > 0", "low", "x must be positive")],
        evidences=[
            EvidenceUnit(
                doc_id="doc1",
                file_or_url=Path("/data/file.txt"),
                segment={"text": "supporting text", "type": "match", "line_number": 10},
                score=0.9,
                extracted_at=datetime(2025, 1, 1),
            )
        ],
        confidence=0.85,
        abstraction_level=AbstractionLevel.PRINCIPLE,
        landmark_potential=0.6,
        hotness=0.4,
        lifecycle=Lifecycle.STABLE,
        create_time=datetime(2025, 1, 1),
        last_modified=datetime(2025, 1, 2),
    )

    print(cluster.to_dict())

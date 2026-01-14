from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

"""
Cognition Layer: Cognitive Graph based on Knowledge Clusters, enabling rich semantic relationships, cognitive navigation, and self-evolution.
"""


class RichEdgeType(Enum):
    """
    Types of semantic edges connecting knowledge clusters in the Cognitive Graph, Directed edge.
    """

    PATHWAY = (
        "pathway"  # Directed edge suggesting a procedural route to another cluster
    )
    BARRIER = "barrier"  # Conditional edge representing a constraint or risk
    ANALOGY = "analogy"  # Undirected edge indicating a conceptual similarity
    SHORTCUT = "shortcut"  # Directed edge representing an expedited path under certain conditions
    RESOLUTION = "resolution"  # Directed edge indicating a solution to a conflict or contradiction


@dataclass
class RichSemanticEdge:
    """
    A rich, executable semantic relationship from current cluster to another.

    Mirrors the edge structure used in Cognition Layer, enabling direct promotion.

    Usage:
        semantic_edges: Dict[str, List[RichSemanticEdge]]

    Examples:
        {
            "pathway": [
                RichSemanticEdge(
                    target_cluster_id="C1002",  # Quantization
                    edge_type="pathway",
                    score=0.95,
                    meta={
                        "steps": ["apply_4bit_quantization", "calibrate_activations"],
                        "required_context": {"task": "finetune"}
                    }
                ),
                RichSemanticEdge(
                    target_cluster_id="C1008",  # Low-Rank Update
                    edge_type="pathway",
                    score=0.9,
                    meta={
                        "steps": ["decompose_delta_w", "train_A_and_B"],
                        "required_context": {}
                    }
                )
            ],
            "barrier": [
                RichSemanticEdge(
                    target_cluster_id="C1005",  # Token Pruning
                    edge_type="barrier",
                    score=0.8,
                    meta={
                        "condition": "quant_bits < 4",
                        "severity": "high",
                        "description": "Ultra-low-bit quantization amplifies pruning noise"
                    }
                )
            ],
            "analogy": [
                RichSemanticEdge(
                    target_cluster_id="C2010",  # Sparse RL Policy Update
                    edge_type="analogy",
                    score=0.85,
                    meta={
                        "source_role": "low_rank_delta_in_lm",
                        "target_role": "sparse_delta_in_policy",
                        "mapping_rules": [
                            "both_constrain_update_space",
                            "freeze_backbone_parameters",
                            "small_trainable_adapter"
                        ]
                    }
                )
            ],
            "shortcut": [
                RichSemanticEdge(
                    target_cluster_id="C1006",  # Mobile Deployment
                    edge_type="shortcut",
                    score=0.9,
                    meta={
                        "trigger_pattern": r".*mobile.*qlora.*",
                        "bypass_steps": 2,
                        "source": "user_query_log"
                    }
                )
            ]
        }
    """

    target_cluster_id: str  # ID of the destination cluster (e.g., "C1002")
    edge_type: RichEdgeType  # Type of the edge (see RichEdgeType class)
    meta: Dict[str, Any]  # Edge-specific payload (see examples below)
    created_at: Optional[str] = None  # ISO 8601 timestamp; optional for immutability
    score: Optional[float] = (
        None  # Confidence or relevance score, normalized [0.0, 1.0] TODO: can be learned dynamically
    )

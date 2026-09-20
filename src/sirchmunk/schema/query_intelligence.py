# Copyright (c) ModelScope Contributors. All rights reserved.
"""Structured query intelligence for multi-path retrieval.

``QueryIntelligence`` consolidates all query analysis signals — keywords,
intent, data requirements, multi-query reformulations, entity/concept
extraction, and structure hypotheses — into a single object produced by one
merged LLM call.  Downstream retrieval paths consume whichever fields they
need without extra LLM round-trips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class QueryIntelligence:
    """All-in-one query analysis produced by ``_build_query_intelligence``.

    Backward-compatible fields
    --------------------------
    keywords : Dict[str, float]
        Compound + atomic keywords with IDF-like weight (same semantics as
        the legacy ``_probe_keywords`` output).
    alt_keywords : Dict[str, float]
        Cross-lingual alternative keywords.
    intent : str
        ``"lookup"`` | ``"comparison"`` | ``"computation"``.
    complexity : str
        ``"simple"`` | ``"moderate"`` | ``"complex"``.
    data_points : List[str]
        Per-fact data requirements (fed into ``DataRequirements``).
    likely_sources : List[str]
        Probable document/section names.
    formula : Optional[str]
        Computation formula if intent is ``computation``.
    time_period : Optional[str]
        Temporal scope mentioned in the query.
    expected_answer_type : str
        Type constraint for the answer (``"person"``, ``"date"``, …).
    target_slot : str
        Relation granularity the question asks for.
    answer_constraints : List[str]
        Additional constraints on the answer form.
    hop_type : str
        ``"single"`` | ``"bridge"`` | ``"comparison"``.

    New multi-path signals
    ----------------------
    reformulations : List[str]
        3–5 diverse restatements of the query for multi-query retrieval.
    entities : List[str]
        Named entities (person, org, place, work) for exact-phrase search.
    concepts : List[str]
        Abstract topic terms for broad recall.
    expected_doc_type : str
        Predicted document genre (``"wiki"``, ``"report"``, ``"article"``,
        ``"code"``, ``"data"``, ``"general"``).
    expected_sections : List[str]
        Section titles likely to contain the answer (for tree/TOC matching).
    location_hint : str
        Where in a document the answer likely sits (``"heading"``,
        ``"table"``, ``"body"``, ``"beginning"``, ``"any"``).
    multi_source_score : float
        Probability that the answer needs multiple document sections.
    """

    # -- Existing keyword/intent signals (backward compatible) --
    keywords: Dict[str, float] = field(default_factory=dict)
    alt_keywords: Dict[str, float] = field(default_factory=dict)
    intent: str = "lookup"
    complexity: str = "simple"

    # -- DataRequirements fields (will be assembled into DataRequirements) --
    data_points: List[str] = field(default_factory=list)
    likely_sources: List[str] = field(default_factory=list)
    formula: Optional[str] = None
    time_period: Optional[str] = None
    expected_answer_type: str = ""
    target_slot: str = ""
    answer_constraints: List[str] = field(default_factory=list)
    hop_type: str = "single"

    # -- NEW: multi-query reformulations --
    reformulations: List[str] = field(default_factory=list)

    # -- NEW: entity/concept extraction --
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)

    # -- NEW: structure hypothesis --
    expected_doc_type: str = "general"
    expected_sections: List[str] = field(default_factory=list)
    location_hint: str = "any"
    multi_source_score: float = 0.0

"""
Mock API endpoints for search functionality
Provides intelligent search suggestions with advanced matching and reranking
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import random

router = APIRouter(prefix="/api/v1/search", tags=["search"])

# Mock knowledge bases for search context
knowledge_bases = [
    {"name": "ai_textbook", "is_default": True},
    {"name": "python_docs", "is_default": False},
    {"name": "research_papers", "is_default": False}
]

@router.get("/{kb_name}/suggestions")
async def get_search_suggestions(kb_name: str, query: str, limit: int = 8):
    """Get search suggestions for file names based on partial query with advanced matching"""
    # Verify knowledge base exists
    kb = next((kb for kb in knowledge_bases if kb["name"] == kb_name), None)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if not query or len(query.strip()) < 2:
        return {
            "success": True,
            "data": [],
            "query": query
        }

    # Mock file suggestions based on common academic file patterns
    mock_files = [
        "machine_learning_fundamentals.pdf",
        "deep_learning_architectures.pdf",
        "neural_networks_theory.pdf",
        "artificial_intelligence_overview.pdf",
        "python_programming_guide.pdf",
        "data_structures_algorithms.pdf",
        "computer_vision_basics.pdf",
        "natural_language_processing.pdf",
        "reinforcement_learning.pdf",
        "statistical_methods.pdf",
        "linear_algebra_concepts.pdf",
        "calculus_applications.pdf",
        "probability_theory.pdf",
        "optimization_techniques.pdf",
        "research_methodology.pdf",
        "academic_writing_guide.pdf",
        "thesis_template.docx",
        "presentation_slides.pptx",
        "dataset_analysis.csv",
        "experiment_results.xlsx"
    ]

    query_lower = query.lower().strip()
    query_words = [word.strip() for word in query_lower.split() if word.strip()]

    # Advanced matching with multi-word support and reranking
    matching_files = []
    for filename in mock_files:
        filename_lower = filename.lower()
        filename_clean = filename.replace("_", " ").replace(".pdf", "").replace(".docx", "").replace(".pptx", "").replace(".csv", "").replace(".xlsx", "")
        filename_clean_lower = filename_clean.lower()
        
        # Check if all query words are present (either in original or cleaned filename)
        word_matches = []
        total_match_score = 0
        
        for word in query_words:
            # Try to find word in both original filename and cleaned version
            match_in_original = filename_lower.find(word)
            match_in_clean = filename_clean_lower.find(word)
            
            if match_in_original >= 0 or match_in_clean >= 0:
                # Use the better match (earlier position gets higher score)
                if match_in_original >= 0 and match_in_clean >= 0:
                    match_pos = min(match_in_original, match_in_clean)
                elif match_in_original >= 0:
                    match_pos = match_in_original
                else:
                    match_pos = match_in_clean
                
                word_matches.append({
                    "word": word,
                    "position": match_pos,
                    "length": len(word)
                })
                
                # Score based on position (earlier = better) and word length
                position_score = max(0, 50 - match_pos)  # Earlier positions get higher scores
                length_score = len(word) * 5  # Longer words get higher scores
                total_match_score += position_score + length_score
            else:
                # Word not found, skip this file
                word_matches = []
                break
        
        # Only include files where all words match
        if len(word_matches) == len(query_words):
            # Calculate additional scoring factors
            exact_phrase_bonus = 0
            if len(query_words) > 1:
                # Check if words appear as a phrase (with reasonable gaps)
                full_query_match = filename_lower.find(query_lower.replace(" ", "_"))
                if full_query_match >= 0:
                    exact_phrase_bonus = 30  # Bonus for exact phrase match
                elif filename_clean_lower.find(query_lower) >= 0:
                    exact_phrase_bonus = 25  # Bonus for phrase match in cleaned version
            
            # Word order bonus - if words appear in the same order as query
            order_bonus = 0
            if len(word_matches) > 1:
                positions = [match["position"] for match in word_matches]
                if positions == sorted(positions):
                    order_bonus = 15  # Bonus for correct word order
            
            # Calculate coverage score - how much of the filename is matched
            total_matched_chars = sum(match["length"] for match in word_matches)
            coverage_score = (total_matched_chars / len(filename_clean)) * 20
            
            # Final relevance score
            relevance = total_match_score + exact_phrase_bonus + order_bonus + coverage_score
            
            # Find the best highlight range (for the first/most important match)
            if query_words:
                # For multi-word queries, highlight the first word or the phrase if found
                if len(query_words) == 1:
                    highlight_start = word_matches[0]["position"]
                    highlight_end = highlight_start + word_matches[0]["length"]
                else:
                    # Try to highlight the full phrase if possible
                    phrase_match = filename_clean_lower.find(query_lower)
                    if phrase_match >= 0:
                        highlight_start = phrase_match
                        highlight_end = phrase_match + len(query_lower)
                    else:
                        # Highlight the first word
                        highlight_start = word_matches[0]["position"]
                        highlight_end = highlight_start + word_matches[0]["length"]
            else:
                highlight_start = 0
                highlight_end = 0

            matching_files.append({
                "filename": filename,
                "display_name": filename_clean.title(),
                "type": filename.split(".")[-1].upper(),
                "size": f"{random.randint(100, 2000)}KB",
                "relevance": relevance,
                "kb_name": kb_name,
                "highlight_start": highlight_start,
                "highlight_end": highlight_end,
                "matched_words": len(word_matches),
                "total_words": len(query_words)
            })

    # Advanced sorting: first by number of matched words, then by relevance score
    matching_files.sort(key=lambda x: (x["matched_words"], x["relevance"]), reverse=True)
    suggestions = matching_files[:limit]

    return {
        "success": True,
        "data": suggestions,
        "query": query,
        "total_matches": len(matching_files)
    }

@router.get("/knowledge-bases")
async def get_knowledge_bases():
    """Get list of available knowledge bases for search"""
    return {
        "success": True,
        "data": knowledge_bases
    }
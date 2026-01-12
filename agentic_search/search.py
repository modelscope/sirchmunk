# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from agentic_search.base import BaseSearch
from agentic_search.learnings.knowledge_bank import KnowledgeBank
from agentic_search.llm.openai import OpenAIChat
from agentic_search.llm.prompts import QUERY_KEYWORDS_EXTRACTION, SEARCH_RESULT_SUMMARY
from agentic_search.retrieve.text_retriever import GrepRetriever
from agentic_search.schema.knowledge import KnowledgeCluster
from agentic_search.schema.request import ContentItem, ImageURL, Message, Request
from agentic_search.utils.file_utils import get_fast_hash
from agentic_search.utils.utils import (
    KeywordValidation,
    extract_fields,
    log_tf_norm_penalty,
)


class AgenticSearch(BaseSearch):

    def __init__(
        self,
        llm: OpenAIChat,
        work_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.work_path: Path = Path(work_path) if work_path is not None else Path.cwd()

        self.llm: OpenAIChat = llm

        self.grep_retriever: GrepRetriever = GrepRetriever(work_path=self.work_path)

        self.knowledge_bank = KnowledgeBank(llm=self.llm, work_path=self.work_path)

        self.verbose: bool = verbose

    @staticmethod
    def _extract_and_validate_keywords(llm_resp: str) -> dict:
        """
        Extract and validate keywords with IDF scores from LLM response.
        """
        res: Dict[str, float] = {}

        # Extract JSON-like content within <KEYWORDS></KEYWORDS> tags
        tag: str = "KEYWORDS"
        keywords_json: Optional[str, None] = extract_fields(
            content=llm_resp,
            tags=[tag],
        ).get(tag.lower(), None)

        if not keywords_json:
            return res

        # Try to parse as dict format
        try:
            res = json.loads(keywords_json)
        except json.JSONDecodeError:
            try:
                res = ast.literal_eval(keywords_json)
            except Exception as e:
                logger.warning("Failed to parse keywords: {}", e)
                return {}

        # Validate using Pydantic model
        try:
            return KeywordValidation(root=res).model_dump()
        except Exception as e:
            logger.warning("Keyword validation failed: {}", e)
            return {}

    @staticmethod
    def fast_deduplicate_by_content(data: List[dict]):
        """
        Deduplicates results based on content fingerprints.
        Keeps the document with the highest total_score for each unique content.

        Args:
            data: sorted grep results by 'total_score' field.

        Returns:
            deduplicated grep results.
        """
        unique_fingerprints = set()
        deduplicated_results = []

        for item in data:
            path = item["path"]

            # 2. Generate a fast fingerprint instead of full MD5
            fingerprint = get_fast_hash(path)

            # 3. Add to results only if this content hasn't been seen yet
            if fingerprint and fingerprint not in unique_fingerprints:
                unique_fingerprints.add(fingerprint)
                deduplicated_results.append(item)

        return deduplicated_results

    def process_grep_results(
        self, results: List[Dict[str, Any]], keywords_with_idf: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Process grep results to calculate total scores for doc and scores for lines based on keywords with IDF.

        Args:
            results: List of grep result dictionaries.
            keywords_with_idf: Dictionary of keywords with their corresponding IDF scores.

        Returns:
            Processed and sorted list of grep result dictionaries.
        """
        results = [
            res
            for res in results
            if res.get("total_matches", 0) >= len(keywords_with_idf)
        ]

        for grep_res in results:
            keywords_tf_in_doc: Dict[str, int] = {
                k.lower(): 0 for k, v in keywords_with_idf.items()
            }
            matches = grep_res.get("matches", [])
            for match_item in matches:
                keywords_tf_in_line: Dict[str, int] = {
                    k.lower(): 0 for k, v in keywords_with_idf.items()
                }
                submatches = match_item.get("data", {}).get("submatches", [])
                for submatch_item in submatches:
                    hit_word: str = submatch_item["match"]["text"].lower()
                    if hit_word in keywords_tf_in_doc:
                        keywords_tf_in_doc[hit_word] += 1
                    if hit_word in keywords_tf_in_line:
                        keywords_tf_in_line[hit_word] += 1
                match_item_score: float = 0.0
                for w, idf in keywords_with_idf.items():
                    match_item_score += idf * log_tf_norm_penalty(
                        keywords_tf_in_line.get(w.lower(), 0)
                    )
                match_item["score"] = (
                    match_item["score"]
                    * match_item_score
                    * log_tf_norm_penalty(
                        count=len(match_item["data"]["lines"]["text"]),
                        ideal_range=(50, 200),
                    )
                )
            # Calculate total score for current document
            total_score: float = 0.0
            for w, idf in keywords_with_idf.items():
                total_score += idf * log_tf_norm_penalty(
                    keywords_tf_in_doc.get(w.lower(), 0)
                )

            grep_res["total_score"] = total_score
            matches.sort(key=lambda x: x["score"], reverse=True)

        results.sort(key=lambda x: x["total_score"], reverse=True)
        results = self.fast_deduplicate_by_content(results)

        return results

    async def search(
        self,
        query: str,
        search_path: Union[str, Path],
        *,
        images: Optional[list] = None,
        max_depth: Optional[int] = 5,
        top_k_files: Optional[int] = 3,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        verbose: Optional[bool] = True,
        grep_timeout: Optional[float] = 60.0,
    ) -> str:

        # Build request
        text_items: List[ContentItem] = [ContentItem(type="text", text=query)]
        image_items: List[ContentItem] = []
        if images is not None and len(images) > 0:
            # TODO: to be implemented
            logger.warning("Image search is not yet implemented.")
            image_items = [
                ContentItem(
                    type="image_url",
                    image_url=ImageURL(url=image_url),
                )
                for image_url in images
            ]

        request: Request = Request(
            messages=[
                Message(
                    role="user",
                    content=text_items + image_items,
                ),
            ],
        )

        # Get enhanced query keywords with IDF scores
        resp_keywords: str = await self.llm.achat(
            messages=request.to_payload(prompt_template=QUERY_KEYWORDS_EXTRACTION),
            stream=False,
        )
        query_keywords: Dict[str, float] = self._extract_and_validate_keywords(
            resp_keywords
        )
        logger.info("Enhanced query keywords: {}", query_keywords)

        # Get grep results
        grep_results: List[Dict[str, Any]] = await self.grep_retriever.retrieve(
            terms=list(query_keywords.keys()),
            path=search_path,
            logic="or",
            case_sensitive=False,
            whole_word=False,
            literal=False,
            regex=True,
            max_depth=max_depth,
            include=None,
            exclude=["*.pyc", "*.log"],
            file_type=None,
            invert_match=False,
            count_only=False,
            line_number=True,
            with_filename=True,
            rank=True,
            rga_no_cache=False,
            rga_cache_max_blob_len=10000000,
            rga_cache_path=None,
            timeout=grep_timeout,
        )

        # Example: [{"path": "", "matches": [], "lines": [], "total_matches": 20, "total_score": 39.70}, ...]
        grep_results: List[Dict[str, Any]] = self.grep_retriever.merge_results(
            grep_results
        )
        grep_results = self.process_grep_results(
            results=grep_results, keywords_with_idf=query_keywords
        )
        if verbose:
            tmp_sep = "\n"
            logger.info(
                f"Grep retrieved files:\n{tmp_sep.join([str(r['path']) for r in grep_results[:top_k_files]])}"
            )

        # Build knowledge cluster
        if verbose:
            logger.info("Building knowledge cluster...")
        cluster: KnowledgeCluster = await self.knowledge_bank.build(
            request=request,
            retrieved_infos=grep_results,
            keywords=query_keywords,
            top_k_files=top_k_files,
            top_k_snippets=5,
            verbose=verbose,
        )

        if cluster is None:
            return f"No relevant information found for the query: {query}"

        # self.knowledge_bank.update(cluster=cluster)
        # self.knowledge_bank.save(cluster=cluster)

        if self.verbose:
            logger.info(json.dumps(cluster.to_dict(), ensure_ascii=False, indent=2))

        # return f"{cluster.name}\n{cluster.description}"

        sep: str = "\n"
        cluster_text_content: str = (
            f"{cluster.name}\n\n"
            f"{sep.join(cluster.description)}\n\n"
            f"{cluster.content if isinstance(cluster.content, str) else sep.join(cluster.content)}"
        )

        result_sum_prompt: str = SEARCH_RESULT_SUMMARY.format(
            user_input=request.get_user_input(),
            text_content=cluster_text_content,
        )

        logger.info("Generating search result summary...")
        search_result: str = await self.llm.achat(
            messages=[{"role": "user", "content": result_sum_prompt}],
            stream=True,
        )

        return search_result

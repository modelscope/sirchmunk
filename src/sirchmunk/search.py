# Copyright (c) ModelScope Contributors. All rights reserved.
import ast
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from sirchmunk.base import BaseSearch

logger = logging.getLogger(__name__)
from sirchmunk.learnings.knowledge_base import KnowledgeBase
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import (
    generate_keyword_extraction_prompt,
    SEARCH_RESULT_SUMMARY,
)
from sirchmunk.retrieve.text_retriever import GrepRetriever
from sirchmunk.schema.knowledge import KnowledgeCluster
from sirchmunk.schema.request import ContentItem, ImageURL, Message, Request
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.utils.constants import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME, SIRCHMUNK_WORK_PATH
from sirchmunk.utils.deps import check_dependencies
from sirchmunk.utils.file_utils import get_fast_hash
from sirchmunk.utils import create_logger, LogCallback
from sirchmunk.utils.install_rga import install_rga
from sirchmunk.utils.utils import (
    KeywordValidation,
    extract_fields,
    log_tf_norm_penalty,
)


class AgenticSearch(BaseSearch):

    def __init__(
        self,
        llm: Optional[OpenAIChat] = None,
        work_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        log_callback: LogCallback = None,
        reuse_knowledge: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        work_path = work_path or SIRCHMUNK_WORK_PATH
        # Ensure path is expanded (handle ~ and environment variables)
        self.work_path: Path = Path(work_path).expanduser().resolve()

        self.llm: OpenAIChat = llm or OpenAIChat(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_MODEL_NAME,
            log_callback=log_callback,
        )

        self.grep_retriever: GrepRetriever = GrepRetriever(work_path=self.work_path)

        # Create bound logger with callback - returns AsyncLogger instance
        self._logger = create_logger(log_callback=log_callback, enable_async=True)

        # Pass log_callback to KnowledgeBase so it can also log through the same callback
        self.knowledge_base = KnowledgeBase(
            llm=self.llm,
            work_path=self.work_path,
            log_callback=log_callback
        )

        # Initialize KnowledgeManager for persistent storage
        self.knowledge_manager = KnowledgeStorage(work_path=str(self.work_path))
        
        # Load historical knowledge clusters from cache
        self._load_historical_knowledge()

        self.verbose: bool = verbose

        self.llm_usages: List[Dict[str, Any]] = []

        # Maximum number of queries to keep per cluster (FIFO strategy)
        self.max_queries_per_cluster: int = 5

        # Initialize embedding client for cluster reuse
        self.embedding_client = None
        # Similarity threshold for cluster reuse
        self.cluster_sim_threshold: float = kwargs.pop('cluster_sim_threshold', 0.85)
        self.cluster_sim_top_k: int = kwargs.pop('cluster_sim_top_k', 3)
        if reuse_knowledge:
            try:
                from sirchmunk.utils.embedding_util import EmbeddingUtil
                
                self.embedding_client = EmbeddingUtil(
                    cache_dir=str(self.work_path / ".cache" / "models")
                )
                logger.debug(
                    f"Embedding client initialized: {self.embedding_client.get_model_info()}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize embedding client: {e}. Cluster reuse disabled."
                )
                self.embedding_client = None

        if not check_dependencies():
            print("Installing rga (ripgrep-all) and rg (ripgrep)...", flush=True)
            install_rga()
    
    def _load_historical_knowledge(self):
        """Load historical knowledge clusters from local cache"""
        try:
            stats = self.knowledge_manager.get_stats()
            cluster_count = stats.get('custom_stats', {}).get('total_clusters', 0)
            # Use sync logger for initialization
            print(f"Loaded {cluster_count} historical knowledge clusters from cache")
        except Exception as e:
            print(f"[WARNING] Failed to load historical knowledge: {e}")
    
    async def _try_reuse_cluster(
        self, 
        query: str, 
        return_cluster: bool = False
    ) -> Optional[Union[str, KnowledgeCluster]]:
        """
        Try to reuse existing knowledge cluster based on semantic similarity.
        
        Args:
            query: Search query string
            return_cluster: Whether to return the full cluster object or just content string
        
        Returns:
            Cluster content string or KnowledgeCluster object if found, None otherwise
        """
        if not self.embedding_client:
            return None
        
        try:
            await self._logger.info("Searching for similar knowledge clusters...")
            
            # Compute query embedding
            query_embedding = (await self.embedding_client.embed([query]))[0]
            
            # Search for similar clusters
            similar_clusters = await self.knowledge_manager.search_similar_clusters(
                query_embedding=query_embedding,
                top_k=self.cluster_sim_top_k,
                similarity_threshold=self.cluster_sim_threshold,
            )
            
            if not similar_clusters:
                await self._logger.info("No similar clusters found, performing new search...")
                return None
            
            # Found similar cluster - process reuse
            best_match = similar_clusters[0]
            await self._logger.success(
                f"♻️ Found similar cluster: {best_match['name']} "
                f"(similarity: {best_match['similarity']:.3f})"
            )
            
            # Retrieve full cluster object
            existing_cluster = await self.knowledge_manager.get(best_match["id"])
            
            if not existing_cluster:
                await self._logger.warning("Failed to retrieve cluster, falling back to new search")
                return None
            
            # Add current query to queries list with FIFO strategy
            self._add_query_to_cluster(existing_cluster, query)
            
            # Update hotness and timestamp for reused cluster
            existing_cluster.hotness = min(1.0, (existing_cluster.hotness or 0.5) + 0.1)
            existing_cluster.last_modified = datetime.now()
            
            # Recompute embedding with new query (before update to avoid double save)
            if self.embedding_client:
                try:
                    from sirchmunk.utils.embedding_util import compute_text_hash
                    
                    combined_text = self.knowledge_manager.combine_cluster_fields(
                        existing_cluster.queries
                    )
                    text_hash = compute_text_hash(combined_text)
                    embedding_vector = (await self.embedding_client.embed([combined_text]))[0]
                    
                    # Update embedding fields in database without triggering save
                    self.knowledge_manager.db.execute(
                        f"""
                        UPDATE {self.knowledge_manager.table_name}
                        SET 
                            embedding_vector = ?::FLOAT[384],
                            embedding_model = ?,
                            embedding_timestamp = CURRENT_TIMESTAMP,
                            embedding_text_hash = ?
                        WHERE id = ?
                        """,
                        [embedding_vector, self.embedding_client.model_id, text_hash, existing_cluster.id]
                    )
                    await self._logger.debug(f"Updated embedding for cluster {existing_cluster.id}")
                except Exception as emb_error:
                    await self._logger.warning(f"Failed to update embedding: {emb_error}")
            
            # Single update call - saves cluster data and embedding together
            await self.knowledge_manager.update(existing_cluster)
            
            await self._logger.success("Reused existing knowledge cluster")
            
            # Return based on return_cluster flag
            if return_cluster:
                return existing_cluster
            else:
                # Format and return cluster content as string
                content = existing_cluster.content
                if isinstance(content, list):
                    content = "\n".join(content)
                return str(content) if content else "Knowledge cluster found but content is empty"
        
        except Exception as e:
            await self._logger.warning(
                f"Failed to search similar clusters: {e}. Falling back to full search."
            )
            return None
    
    def _add_query_to_cluster(self, cluster: KnowledgeCluster, query: str) -> None:
        """
        Add query to cluster's queries list with FIFO strategy.
        Keeps only the most recent N queries (where N = max_queries_per_cluster).
        
        Args:
            cluster: KnowledgeCluster to update
            query: New query to add
        """
        # Add query if not already present
        if query not in cluster.queries:
            cluster.queries.append(query)
        
        # Apply FIFO strategy: keep only the most recent N queries
        if len(cluster.queries) > self.max_queries_per_cluster:
            # Remove oldest queries (from the beginning)
            cluster.queries = cluster.queries[-self.max_queries_per_cluster:]
    
    async def _save_cluster_with_embedding(self, cluster: KnowledgeCluster) -> None:
        """
        Save knowledge cluster to persistent storage and compute embedding.
        
        Args:
            cluster: KnowledgeCluster to save
        """
        # Save knowledge cluster to persistent storage
        try:
            await self.knowledge_manager.insert(cluster)
            await self._logger.info(f"Saved knowledge cluster {cluster.id} to cache")
        except Exception as e:
            # If cluster exists, update it instead
            try:
                await self.knowledge_manager.update(cluster)
                await self._logger.info(f"Updated knowledge cluster {cluster.id} in cache")
            except Exception as update_error:
                await self._logger.warning(f"Failed to save knowledge cluster: {update_error}")
                return
        
        # Compute and store embedding for the cluster
        if self.embedding_client:
            try:
                from sirchmunk.utils.embedding_util import compute_text_hash
                
                # Combine queries for embedding
                combined_text = self.knowledge_manager.combine_cluster_fields(
                    cluster.queries
                )
                text_hash = compute_text_hash(combined_text)
                
                # Compute embedding
                embedding_vector = (await self.embedding_client.embed([combined_text]))[0]
                
                # Store embedding
                await self.knowledge_manager.store_embedding(
                    cluster_id=cluster.id,
                    embedding_vector=embedding_vector,
                    embedding_model=self.embedding_client.model_id,
                    embedding_text_hash=text_hash
                )
                
                await self._logger.debug(f"Computed and stored embedding for cluster {cluster.id}")
            
            except Exception as e:
                await self._logger.warning(f"Failed to compute embedding for cluster {cluster.id}: {e}")
    
    async def _search_by_filename(
        self,
        query: str,
        search_paths: Union[str, Path, List[str], List[Path]],
        max_depth: Optional[int] = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        grep_timeout: Optional[float] = 60.0,
        top_k: Optional[int] = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform filename-only search without LLM keyword extraction.
        
        Args:
            query: Search query (used as filename pattern)
            search_paths: Paths to search in
            max_depth: Maximum directory depth
            include: File patterns to include
            exclude: File patterns to exclude
            grep_timeout: Timeout for grep operations
            top_k: Maximum number of results to return
        
        Returns:
            List of file matches with metadata
        """
        await self._logger.info("Performing filename-only search...")
        
        # Extract potential filename patterns from query
        patterns = []
        
        # Check if query looks like a file pattern (contains file extensions or wildcards)
        if any(char in query for char in ['*', '?', '[', ']']):
            # Treat as direct glob/regex pattern
            patterns = [query]
            await self._logger.info(f"Using direct pattern: {query}")
        else:
            # Split into words and create flexible patterns
            words = [w.strip() for w in query.strip().split() if w.strip()]
            
            if not words:
                await self._logger.warning("No valid words in query")
                return []
            
            # Strategy: Create patterns for each word that match anywhere in filename
            # Use non-greedy matching and case-insensitive by default
            for word in words:
                # Escape special regex characters in the word
                escaped_word = re.escape(word)
                # Match word anywhere in filename (case-insensitive handled in retrieve_by_filename)
                pattern = f".*{escaped_word}.*"
                patterns.append(pattern)
                await self._logger.debug(f"Created pattern for word '{word}': {pattern}")
        
        if not patterns:
            await self._logger.warning("No valid filename patterns extracted from query")
            return []
        
        await self._logger.info(f"Searching with {len(patterns)} pattern(s): {patterns}")
        
        try:
            # Use GrepRetriever's filename search
            await self._logger.debug(f"Calling retrieve_by_filename with {len(patterns)} patterns")
            results = await self.grep_retriever.retrieve_by_filename(
                patterns=patterns,
                path=search_paths,
                case_sensitive=False,
                max_depth=max_depth,
                include=include,
                exclude=exclude or ["*.pyc", "*.log"],
                timeout=grep_timeout,
            )
            
            if results:
                results = results[:top_k]
                await self._logger.success(f" ✓ Found {len(results)} matching files", flush=True)
            else:
                await self._logger.warning("No files matched the patterns")
            
            return results
        
        except Exception as e:
            await self._logger.error(f"Filename search failed: {e}")
            import traceback
            await self._logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    @staticmethod
    def _parse_summary_response(llm_response: str) -> tuple[str, bool]:
        """
        Parse LLM response to extract summary and save decision.
        
        Args:
            llm_response: Raw LLM response containing SUMMARY and SHOULD_SAVE tags
        
        Returns:
            Tuple of (summary_text, should_save_flag)
        """
        # Extract SUMMARY content
        summary_fields = extract_fields(content=llm_response, tags=["SUMMARY", "SHOULD_SAVE"])
        
        summary = summary_fields.get("summary", "").strip()
        should_save_str = summary_fields.get("should_save", "true").strip().lower()
        
        # Parse should_save flag
        should_save = should_save_str in ["true", "yes", "1"]
        
        # If extraction failed, use entire response as summary and assume should save
        if not summary:
            summary = llm_response.strip()
            should_save = True
        
        return summary, should_save

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
                return {}

        # Validate using Pydantic model
        try:
            return KeywordValidation(root=res).model_dump()
        except Exception as e:
            return {}

    @staticmethod
    def _extract_and_validate_multi_level_keywords(
        llm_resp: str,
        num_levels: int = 3
    ) -> List[Dict[str, float]]:
        """
        Extract and validate multiple sets of keywords from LLM response.

        Args:
            llm_resp: LLM response containing keyword sets
            num_levels: Number of keyword granularity levels to extract

        Returns:
            List of keyword dicts, one for each level: [level1_keywords, level2_keywords, ...]
        """
        keyword_sets: List[Dict[str, float]] = []

        # Generate tags dynamically based on num_levels
        tags = [f"KEYWORDS_LEVEL_{i+1}" for i in range(num_levels)]

        # Extract all fields at once
        extracted_fields = extract_fields(content=llm_resp, tags=tags)

        for level_idx, tag in enumerate(tags, start=1):
            keywords_dict: Dict[str, float] = {}
            keywords_json: Optional[str] = extracted_fields.get(tag.lower(), None)

            if not keywords_json:
                keyword_sets.append({})
                continue

            # Try to parse as dict format
            try:
                keywords_dict = json.loads(keywords_json)
            except json.JSONDecodeError:
                try:
                    keywords_dict = ast.literal_eval(keywords_json)
                except Exception as e:
                    keyword_sets.append({})
                    continue

            # Validate using Pydantic model
            try:
                validated = KeywordValidation(root=keywords_dict).model_dump()
                keyword_sets.append(validated)
            except Exception as e:
                keyword_sets.append({})

        return keyword_sets

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
        search_paths: Union[str, Path, List[str], List[Path]],
        *,
        mode: Literal["DEEP", "FILENAME_ONLY"] = "DEEP",
        images: Optional[list] = None,
        max_depth: Optional[int] = 5,
        top_k_files: Optional[int] = 3,
        keyword_levels: Optional[int] = 3,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        verbose: Optional[bool] = True,
        grep_timeout: Optional[float] = 60.0,
        return_cluster: Optional[bool] = False,
    ) -> Union[str, List[Dict[str, Any]], KnowledgeCluster]:
        """
        Perform intelligent search with multi-level keyword extraction.

        Args:
            query: Search query string
            search_paths: Paths to search in
            mode: Search mode (DEEP/FILENAME_ONLY), default is DEEP
            images: Optional image inputs
            max_depth: Maximum directory depth to search
            top_k_files: Number of top files to grep-retrieve
            keyword_levels: Number of keyword granularity levels (default: 3)
                          - Higher values provide more fallback options
                          - Recommended: 3-5 levels
            include: File patterns to include
            exclude: File patterns to exclude
            verbose: Enable verbose logging
            grep_timeout: Timeout for grep operations
            return_cluster: Whether to return the full knowledge cluster. Ignore if mode is `FILENAME_ONLY`.

        Mode behaviors:
            - In FILENAME_ONLY mode, performs fast filename search without LLM involvement. Returns list of matching files.
               Format: {'filename': 'Attention_Is_All_You_Need.pdf', 'match_score': 0.8, 'matched_pattern': '.*Attention.*', 'path': '/path/to/Attention_Is_All_You_Need.pdf', 'type': 'filename_match'}

            +--------------+------------------+-----------------------+------------------------+
            | Feature      | FILENAME_ONLY    | FAST (To be designed) | DEEP (Current)         |
            +--------------+------------------+-----------------------+------------------------+
            | Speed        | Very Fast (<1s)  | Fast (<5s)           | Slow (5-30s)          |
            | LLM Calls    | 0 times          | 1-2 times             | 4-5 times              |
            | Return Type  | List[Dict]       | str / Cluster         | str / Cluster          |
            | Use Case     | File Location    | Rapid Content Search  | Deep Knowledge Extract |
            +--------------+------------------+-----------------------+------------------------+

        Returns:
            Search result summary string, or KnowledgeCluster if return_cluster is True, or List[Dict[str, Any]] for FILENAME_ONLY mode.
        """
        # Handle FILENAME_ONLY mode: fast filename search without LLM
        if mode == "FILENAME_ONLY":
            filename_results: List[Dict[str, Any]] = await self._search_by_filename(
                query=query,
                search_paths=search_paths,
                max_depth=max_depth,
                include=include,
                exclude=exclude,
                grep_timeout=grep_timeout,
                top_k=top_k_files,
            )

            if not filename_results:
                error_msg = f"No files found matching query: '{query}'"
                await self._logger.warning(error_msg)
                return None if return_cluster else error_msg

            await self._logger.success(f"Retrieved {len(filename_results)} matching files")

            return filename_results

        # Try to reuse existing cluster based on semantic similarity
        reused_result = await self._try_reuse_cluster(query, return_cluster=return_cluster)
        if reused_result:
            return reused_result

        # Build request
        text_items: List[ContentItem] = [ContentItem(type="text", text=query)]
        image_items: List[ContentItem] = []
        if images is not None and len(images) > 0:
            # TODO: to be implemented
            await self._logger.warning("Image search is not yet implemented.")
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

        # Extract multi-level keywords in one LLM call
        await self._logger.info(f"Extracting {keyword_levels}-level query keywords.")

        # Generate dynamic prompt based on keyword_levels
        dynamic_prompt = generate_keyword_extraction_prompt(num_levels=keyword_levels)
        keyword_extraction_prompt = dynamic_prompt.format(user_input=request.get_user_input())

        resp_keywords_response = await self.llm.achat(
            messages=[{"role": "user", "content": keyword_extraction_prompt}],
            stream=False,
        )
        resp_keywords: str = resp_keywords_response.content
        self.llm_usages.append(resp_keywords_response.usage)
        
        await self._logger.success(" ✓", flush=True)

        # Parse N sets of keywords
        keyword_sets: List[Dict[str, float]] = self._extract_and_validate_multi_level_keywords(
            resp_keywords,
            num_levels=keyword_levels
        )

        # Ensure we have keyword_levels sets (even if some are empty)
        while len(keyword_sets) < keyword_levels:
            keyword_sets.append({})

        # Log all extracted keyword sets
        for level_idx, keywords in enumerate(keyword_sets, start=1):
            specificity = "General" if level_idx == 1 else "Specific" if level_idx == keyword_levels else f"Level {level_idx}"
            await self._logger.info(f"Level {level_idx} ({specificity}) keywords: {keywords}")

        # Try each keyword set in order (from general to specific) until we get results
        # Using priority hit principle: stop as soon as we find results
        grep_results: List[Dict[str, Any]] = []
        query_keywords: Dict[str, float] = {}

        for level_idx, keywords in enumerate(keyword_sets, start=1):
            if not keywords:
                await self._logger.warning(f"Level {level_idx} keywords set is empty, skipping...")
                continue

            specificity = "General" if level_idx == 1 else "Specific" if level_idx == keyword_levels else f"Level {level_idx}"
            await self._logger.info(f"Searching with Level {level_idx} ({specificity}) keywords.")

            # Perform grep search with current keyword set
            temp_grep_results: List[Dict[str, Any]] = await self.grep_retriever.retrieve(
                terms=list(keywords.keys()),
                path=search_paths,
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

            # Merge and process results
            temp_grep_results = self.grep_retriever.merge_results(temp_grep_results)
            temp_grep_results = self.process_grep_results(
                results=temp_grep_results, keywords_with_idf=keywords
            )

            # Check if we found results
            if len(temp_grep_results) > 0:
                await self._logger.success(f" ✓ (found {len(temp_grep_results)} files)", flush=True)
                grep_results = temp_grep_results
                query_keywords = keywords
                break
            else:
                await self._logger.warning(" ✗ (no results, trying next level)", flush=True)

        # If still no results after all attempts
        if len(grep_results) == 0:
            await self._logger.error(f"All {keyword_levels} keyword granularity levels failed to find results")

        if verbose:
            tmp_sep = "\n"
            file_list = [str(r['path']) for r in grep_results[:top_k_files]]
            await self._logger.info(f"Found {len(grep_results)} files, top {len(file_list)}:\n{tmp_sep.join(file_list)}")

        if len(grep_results) == 0:
            error_msg = f"No relevant information found for the query: {query}"
            return None if return_cluster else error_msg

        # Build knowledge cluster
        await self._logger.info("Building knowledge cluster...")
        cluster: KnowledgeCluster = await self.knowledge_base.build(
            request=request,
            retrieved_infos=grep_results,
            keywords=query_keywords,
            top_k_files=top_k_files,
            top_k_snippets=5,
            verbose=verbose,
        )

        self.llm_usages.extend(self.knowledge_base.llm_usages)
        
        await self._logger.success(" ✓", flush=True)

        if cluster is None:
            error_msg = f"No relevant information found for the query: {query}"
            return None if return_cluster else error_msg

        if self.verbose:
            await self._logger.info(json.dumps(cluster.to_dict(), ensure_ascii=False, indent=2))

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

        await self._logger.info("Generating search result summary...")
        search_result_response = await self.llm.achat(
            messages=[{"role": "user", "content": result_sum_prompt}],
            stream=True,
        )
        llm_response: str = search_result_response.content
        self.llm_usages.append(search_result_response.usage)
        await self._logger.success(" ✓", flush=True)
        await self._logger.success("Search completed successfully!")

        # Parse LLM response to extract summary and save decision
        search_result, should_save = self._parse_summary_response(llm_response)

        # Add search results (file paths) to the cluster
        if grep_results:
            cluster.search_results.append(search_result)
        
        # Add current query to queries list with FIFO strategy
        self._add_query_to_cluster(cluster, query)

        # Save cluster based on LLM's quality evaluation
        if should_save:
            await self._save_cluster_with_embedding(cluster)
        else:
            await self._logger.info(
                "Cluster not saved - LLM determined insufficient quality or relevance"
            )

        return cluster if return_cluster else search_result

# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from agentic_search.base import BaseSearch
from agentic_search.learnings.knowledge_bank import KnowledgeBank
from agentic_search.llm.openai import OpenAIChat
from agentic_search.llm.prompts import QUERY_KEYWORDS_EXTRACTION, SEARCH_RESULT_SUMMARY
from agentic_search.retrieve.text_retriever import GrepRetriever
from agentic_search.schema.request import ContentItem, ImageURL, Message, Request


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

        self.grep_retriever: GrepRetriever = GrepRetriever()

        self.knowledge_bank = KnowledgeBank(llm=self.llm, work_path=self.work_path)

        self.verbose: bool = verbose

    def search(
        self,
        query: str,
        data_folder: Union[str, Path],
        images: Optional[list] = None,
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

        # Get enhanced query keywords
        resp_keywords: str = self.llm.chat(
            messages=request.to_payload(prompt_template=QUERY_KEYWORDS_EXTRACTION),
            stream=True,
        )
        query_keywords: List[str] = [
            w.strip() for w in resp_keywords.split(",") if w.strip()
        ]
        logger.info("Enhanced query keywords: {}", query_keywords)

        # Get grep results
        grep_results: List[Dict[str, Any]] = self.grep_retriever.retrieve(
            terms=query_keywords,
            path=data_folder,
            logic="or",
            case_sensitive=False,
            whole_word=False,
            literal=False,
            regex=True,
            max_depth=3,
            include=None,
            exclude=["*.pyc", "*.log"],
            file_type=None,
            invert_match=False,
            count_only=False,
            line_number=True,
            with_filename=True,
            rank=True,
        )

        grep_results = self.grep_retriever.merge_results(grep_results)
        logger.info("Total grep files: {}", len(grep_results))

        # Build knowledge cluster
        logger.info("Building knowledge cluster...")
        cluster = self.knowledge_bank.build(
            request=request,
            retrieved_infos=grep_results,
            top_k_files=5,
        )

        self.knowledge_bank.update(cluster=cluster)
        self.knowledge_bank.save(cluster=cluster)

        if self.verbose:
            logger.info(json.dumps(cluster.to_dict(), ensure_ascii=False, indent=2))

        # return f"{cluster.name}\n{cluster.description}"

        sep: str = "\n"
        cluster_text_content: str = (f"{cluster.name}\n\n"
                                     f"{sep.join(cluster.description)}\n\n"
                                     f"{cluster.content if isinstance(cluster.content, str) else sep.join(cluster.content)}")

        result_sum_prompt: str = SEARCH_RESULT_SUMMARY.format(
            user_input=request.get_user_query(),
            text_content=cluster_text_content,
        )

        logger.info("Generating search result summary...")
        search_result: str = self.llm.chat(
            messages=[{"role": "user", "content": result_sum_prompt}],
            stream=True,
        )

        return search_result

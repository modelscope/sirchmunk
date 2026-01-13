import re
from typing import List, Optional, Tuple, Union

import numpy as np
from modelscope import snapshot_download

from src.agentic_search.llm.openai import OpenAIChat
from src.agentic_search.llm.prompts import (
    SNAPSHOT_KEYWORDS_EXTRACTION,
    SNAPSHOT_TOC_EXTRACTION,
)


class KeyPhraseExtractor:
    """
    Key phrase extraction using sentence-transformers and scikit-learn.
    """

    def __init__(
        self,
        model_id: str = "AI-ModelScope/all-MiniLM-L6-v2",
        device: Union[str, None] = None,
        stop_words: Optional[List[str]] = None,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        """
        Key phrase extraction using sentence-transformers and scikit-learn.

        Args:
            model_id (str): model id from ModelScope.
            device (str): device to load the model on ('cpu' or 'cuda' or 'mps').
            stop_words (Optional[List[str]]): list of stop words to filter out.
            ngram_range (Tuple[int, int]): n-gram range for candidate generation.
        """
        from sentence_transformers import SentenceTransformer

        model_dir: str = snapshot_download(
            model_id=model_id,
            ignore_patterns=[
                "openvino/*",
                "onnx/*",
                "pytorch_model.bin",
                "rust_model.ot",
                "tf_model.h5",
            ],
        )
        self.model = SentenceTransformer(model_dir, device=device)

        self.ngram_range = ngram_range
        self.stop_words = set(stop_words) if stop_words else set()
        if not self.stop_words:
            self.stop_words = {
                # Chinese stop words
                "的",
                "了",
                "在",
                "是",
                "我",
                "有",
                "和",
                "就",
                "不",
                "人",
                "都",
                "一",
                "一个",
                "上",
                # English stop words
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
            }

    @staticmethod
    def _preprocess(text: str) -> str:
        """
        Keep only Chinese, English, digits, and spaces; convert to lowercase.
        """
        text = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", text.lower())
        return " ".join(text.split())

    def _generate_candidates(self, docs: List[str]) -> List[str]:
        """
        Generate candidate key phrases using CountVectorizer.
        """
        from sklearn.feature_extraction.text import CountVectorizer

        n_docs = len(docs)
        if n_docs == 0:
            return []

        vectorizer_params = {
            "ngram_range": self.ngram_range,
            "stop_words": list(self.stop_words),
            "token_pattern": r"(?u)\b[\w\u4e00-\u9fa5]+\b",
            "lowercase": True,
            "min_df": 1,
        }
        if n_docs > 3:
            vectorizer_params["max_df"] = 0.85

        try:
            vectorizer = CountVectorizer(**vectorizer_params)
            vectorizer.fit(docs)
        except ValueError as e:
            if "max_df corresponds to" in str(e):
                # fallback
                vectorizer_params.pop("max_df", None)
                vectorizer = CountVectorizer(**vectorizer_params)
                vectorizer.fit(docs)
            else:
                raise

        candidates = vectorizer.get_feature_names_out().tolist()
        candidates = [c.strip() for c in candidates if len(re.sub(r"\s+", "", c)) > 1]

        return candidates

    def extract(
        self,
        contents: List[str],
        top_k: int = 20,
        use_mmr: bool = True,
        diversity: float = 0.5,
        candidates: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """
        Extract key phrases from the given contents.

        contents (List[str]): List of document strings.
        top_k (int): Number of top key phrases to return.
        use_mmr (bool): Whether to use MMR for diversity.
        diversity (float): Diversity factor for MMR (0 to 1).
        candidates (Optional[List[str]]): Predefined candidate phrases/keywords.
        batch_size (int): Batch size for encoding.
        """
        if not contents:
            return []

        # Step 1: merge and preprocess documents
        doc = " ".join(contents)
        doc = self._preprocess(doc)

        # Step 2: Generate candidates if not provided
        if candidates is None:
            candidates = self._generate_candidates([doc])
        if not candidates:
            return []

        # Step 3: Encode document and candidates
        sentences = [doc] + candidates
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        doc_emb = embeddings[0]
        cand_embs = embeddings[1:]

        # Step 4: Compute similarities
        similarities = (doc_emb @ cand_embs.T).flatten()  # shape: (n_candidates,)

        if not use_mmr:
            idx = np.argsort(similarities)[::-1][:top_k]
            return [(candidates[i], float(similarities[i])) for i in idx]

        # Step 5 (Optional): MMR diversity selection
        selected_idx = [int(np.argmax(similarities))]
        remaining_idx = list(set(range(len(candidates))) - set(selected_idx))

        while len(selected_idx) < top_k and remaining_idx:
            candidate_scores = []
            for i in remaining_idx:
                sim_to_doc = similarities[i]
                sim_to_selected = np.max(cand_embs[selected_idx] @ cand_embs[i])
                mmr_score = diversity * sim_to_doc - (1 - diversity) * sim_to_selected
                candidate_scores.append((i, mmr_score))

            # Select the candidate with the highest MMR score
            next_i = max(candidate_scores, key=lambda x: x[1])[0]
            selected_idx.append(next_i)
            remaining_idx.remove(next_i)

        return [(candidates[i], float(similarities[i])) for i in selected_idx]


class TextInsights:
    """
    Text insights: key information extraction.
    """

    def __init__(self, llm: Optional[OpenAIChat] = None, **kwargs):
        """
        Initialize TextInsights with an optional LLM instance.

        Args:
            llm (Optional[OpenAIChat]): An instance of OpenAIChat.
            **kwargs: Additional keyword arguments for KeyPhraseExtractor.
        """
        self._llm = llm

        self._key_phrase_extractor = (
            KeyPhraseExtractor(
                model_id="AI-ModelScope/all-MiniLM-L6-v2",
                device=None,
                ngram_range=(1, 2),
            )
            if self._llm is None
            else None
        )

        self._kwargs = kwargs

    def extract_phrase(self, contents: List[str], max_num: int = 20) -> List[str]:
        """
        Extract key phrases from the given contents.

        Args:
            contents (List[str]): List of document strings.
            max_num (int): Maximum number of key phrases to extract.

        Returns:
            List[str]: Extracted key phrases.
        """

        if self._llm is not None:
            prompt = SNAPSHOT_KEYWORDS_EXTRACTION.format(
                document_content="\n\n".join(contents), max_num=max_num
            )
            messages = [{"role": "user", "content": prompt}]
            response: str = self._llm.chat(
                messages=messages,
                stream=True,
            )

            results = [
                phrase.strip().lower()
                for phrase in response.split(",")
                if phrase.strip()
            ]

        else:
            extracted_phrases = self._key_phrase_extractor.extract(
                contents=contents, top_k=max_num, **self._kwargs
            )
            results = [phrase for phrase, _ in extracted_phrases]

        return results

    def extract_toc(self, contents: List[str]) -> str:
        """
        Extract the `Table of contents from input document.`
        """

        if self._llm is None:
            return ""

        prompt = SNAPSHOT_TOC_EXTRACTION.format(document_content="\n\n".join(contents))
        messages = [{"role": "user", "content": prompt}]

        response: str = self._llm.chat(
            messages=messages,
            stream=True,
        )

        return response

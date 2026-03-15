# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unified BM25 scoring utility.

Combines ``bm25s`` (optimised C-backed BM25 engine) with
``TokenizerUtil`` (Qwen sub-word tokeniser for CJK + Latin) into a
single reusable scorer.

Fallback chain
--------------
1. **bm25s + TokenizerUtil** — best quality for multilingual corpora.
2. **bm25s + built-in tokeniser** — good for English-only.
3. **Built-in Okapi BM25 + regex tokeniser** — zero external deps.
"""

import math
import re
from typing import Any, Dict, List, Optional

_CJK_WORD_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")


def _regex_tokenize(texts: List[str]) -> List[List[str]]:
    """CJK-aware regex tokenisation fallback."""
    return [_CJK_WORD_RE.findall(t.lower()) for t in texts]


class BM25Scorer:
    """BM25 scorer with layered tokeniser and engine fallback.

    Parameters
    ----------
    tokenizer : TokenizerUtil | None
        When provided, ``tokenizer.segment()`` produces sub-word tokens
        for both indexing and querying.  This matters for CJK text where
        whitespace splitting is useless.
    """

    def __init__(self, tokenizer: Any = None):
        self._tokenizer = tokenizer
        self._bm25s: Any = None
        self._Tokenized: Any = None
        try:
            import bm25s
            from bm25s.tokenization import Tokenized

            self._bm25s = bm25s
            self._Tokenized = Tokenized
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenise *texts* using the best available tokeniser."""
        if self._tokenizer is not None:
            return [self._tokenizer.segment(t) for t in texts]
        return _regex_tokenize(texts)

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Return a BM25 score for every document w.r.t. *query*."""
        if not documents:
            return []
        if self._bm25s is not None:
            try:
                return self._score_bm25s(query, documents)
            except Exception:
                pass
        return self._score_builtin(query, documents)

    def rerank(
        self, query: str, documents: List[str], top_k: int,
    ) -> Optional[List[int]]:
        """Return original indices of the *top_k* most relevant documents."""
        scores = self.score(query, documents)
        if not scores:
            return None
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in indexed[:top_k]]

    # ------------------------------------------------------------------
    # bm25s engine
    # ------------------------------------------------------------------

    def _score_bm25s(self, query: str, documents: List[str]) -> List[float]:
        bm25s = self._bm25s

        if self._tokenizer is not None:
            all_tokens = self.tokenize(documents + [query])
            doc_tokens = all_tokens[:-1]
            q_tokens = [all_tokens[-1]]
            vocab = self._build_vocab(doc_tokens + q_tokens)
            corpus_tok = self._to_bm25s(doc_tokens, vocab)
            query_tok = self._to_bm25s(q_tokens, vocab)
        else:
            corpus_tok = bm25s.tokenize(documents, stopwords="en")
            query_tok = bm25s.tokenize([query], stopwords="en")

        retriever = bm25s.BM25()
        retriever.index(corpus_tok)

        k = len(documents)
        results, scores = retriever.retrieve(query_tok, k=k)

        score_map: Dict[int, float] = {}
        for i, idx in enumerate(results[0]):
            score_map[int(idx)] = float(scores[0][i])
        return [score_map.get(i, 0.0) for i in range(len(documents))]

    def _to_bm25s(
        self, token_lists: List[List[str]], vocab: Dict[str, int],
    ) -> Any:
        ids = [[vocab[t] for t in tokens if t in vocab] for tokens in token_lists]
        return self._Tokenized(ids=ids, vocab=vocab)

    @staticmethod
    def _build_vocab(token_lists: List[List[str]]) -> Dict[str, int]:
        vocab: Dict[str, int] = {}
        for tokens in token_lists:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        return vocab

    # ------------------------------------------------------------------
    # Built-in Okapi BM25 (no external dependency)
    # ------------------------------------------------------------------

    def _score_builtin(
        self, query: str, documents: List[str],
        k1: float = 1.5, b: float = 0.75,
    ) -> List[float]:
        query_tokens = self.tokenize([query])[0]
        doc_tokens = self.tokenize(documents)

        n = len(doc_tokens)
        if n == 0:
            return []
        avg_dl = sum(len(t) for t in doc_tokens) / n

        df: Dict[str, int] = {}
        for tokens in doc_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        scores: List[float] = []
        for tokens in doc_tokens:
            dl = len(tokens)
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            s = 0.0
            for qt in query_tokens:
                if qt not in df:
                    continue
                idf = math.log((n - df[qt] + 0.5) / (df[qt] + 0.5) + 1)
                tf = tf_map.get(qt, 0)
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
            scores.append(s)
        return scores

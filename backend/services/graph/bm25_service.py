import logging
import math
import re
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class BM25Service:

    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.terms: dict[str, float] = {}
        self.avg_doc_len: float = 0
        self.total_docs: int = 0
        self.k1: float = 1.5
        self.b: float = 0.75

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\u4e00-\u9fff\w]", " ", text)
        tokens = []
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
        words = text.split()
        for w in words:
            if len(w) >= 1:
                tokens.append(w)
        return tokens

    def index_document(self, doc_id: str, text: str, metadata: dict = None):
        tokens = self._tokenize(text)
        self.documents[doc_id] = {
            "tokens": tokens,
            "text": text,
            "metadata": metadata or {},
        }
        self._rebuild_stats()

    def remove_document(self, doc_id: str):
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._rebuild_stats()

    def _rebuild_stats(self):
        self.total_docs = len(self.documents)
        if self.total_docs == 0:
            self.terms = {}
            self.avg_doc_len = 0
            return

        total_len = 0
        doc_freq: dict[str, int] = defaultdict(int)
        for doc in self.documents.values():
            tokens = doc["tokens"]
            total_len += len(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freq[t] += 1

        self.avg_doc_len = total_len / max(self.total_docs, 1)
        self.terms = {}
        for term, df in doc_freq.items():
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
            self.terms[term] = idf

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for doc_id, doc in self.documents.items():
            doc_tokens = doc["tokens"]
            doc_len = len(doc_tokens)
            token_freq = defaultdict(int)
            for t in doc_tokens:
                token_freq[t] += 1

            score = 0.0
            for qt in query_tokens:
                if qt not in self.terms:
                    continue
                idf = self.terms[qt]
                tf = token_freq.get(qt, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
                score += idf * numerator / denominator

            if score > 0:
                scores.append({"id": doc_id, "score": score, "metadata": doc["metadata"]})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def clear(self):
        self.documents.clear()
        self.terms.clear()
        self.total_docs = 0
        self.avg_doc_len = 0

    def count(self) -> int:
        return self.total_docs
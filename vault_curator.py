#!/usr/bin/env python3
"""
vault_curator.py — Core engine for UGOS Neural Vault & Media Curator.
Designed for 100% on-device semantic memory indexing and photo reel clustering on UGREEN AI NAS.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


class NeuralVaultCurator:
    """Local, zero-cloud semantic search and media clustering engine."""

    def __init__(self, n_components: int = 16):
        self.n_components = n_components
        self.items: list[dict[str, Any]] = []
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.vectors: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text)]

    def index_corpus(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Index photo metadata, tags, and descriptions into dense semantic vectors."""
        self.items = items
        if not items:
            return {"status": "empty", "count": 0}

        # 1. Build vocabulary & document term frequency
        doc_tokens = []
        vocab_counts: dict[str, int] = {}
        for item in items:
            text = f"{item.get('title', '')} {item.get('description', '')} {' '.join(item.get('tags', []))}"
            tokens = self._tokenize(text)
            doc_tokens.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                vocab_counts[t] = vocab_counts.get(t, 0) + 1

        self.vocabulary = {t: idx for idx, t in enumerate(sorted(vocab_counts.keys()))}
        n_docs = len(items)
        n_vocab = len(self.vocabulary)
        if n_vocab == 0:
            return {"status": "no_vocabulary", "count": 0}

        # 2. Compute IDF
        self.idf = np.zeros(n_vocab, dtype=np.float32)
        for term, doc_count in vocab_counts.items():
            idx = self.vocabulary[term]
            self.idf[idx] = math.log((1.0 + n_docs) / (1.0 + doc_count)) + 1.0

        # 3. Compute TF-IDF matrix
        tfidf = np.zeros((n_docs, n_vocab), dtype=np.float32)
        for doc_idx, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, count in tf.items():
                if t in self.vocabulary:
                    term_idx = self.vocabulary[t]
                    tfidf[doc_idx, term_idx] = (count / len(tokens)) * self.idf[term_idx]
            norm = np.linalg.norm(tfidf[doc_idx])
            if norm > 1e-6:
                tfidf[doc_idx] /= norm

        # 4. Truncated SVD / LSA dimensionality reduction via numpy SVD
        k = min(self.n_components, n_docs, n_vocab)
        if k > 0 and n_docs > 1:
            u, s, vt = np.linalg.svd(tfidf, full_matrices=False)
            self.components = vt[:k]
            self.vectors = np.dot(tfidf, self.components.T)
        else:
            self.components = np.eye(n_vocab, dtype=np.float32)[:k]
            self.vectors = tfidf[:, :k] if k > 0 else tfidf

        # Normalize dense vectors
        for i in range(len(self.vectors)):
            vnorm = np.linalg.norm(self.vectors[i])
            if vnorm > 1e-6:
                self.vectors[i] /= vnorm

        return {
            "status": "indexed",
            "count": n_docs,
            "vocab_size": n_vocab,
            "dimensions": self.vectors.shape[1] if self.vectors is not None else 0
        }

    def _embed_query(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        n_vocab = len(self.vocabulary)
        q_vec = np.zeros(n_vocab, dtype=np.float32)
        if not tokens or n_vocab == 0 or self.idf is None:
            return q_vec

        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, count in tf.items():
            if t in self.vocabulary:
                term_idx = self.vocabulary[t]
                q_vec[term_idx] = (count / len(tokens)) * self.idf[term_idx]
        norm = np.linalg.norm(q_vec)
        if norm > 1e-6:
            q_vec /= norm

        if self.components is not None:
            dense_q = np.dot(q_vec, self.components.T)
            dnorm = np.linalg.norm(dense_q)
            return dense_q / dnorm if dnorm > 1e-6 else dense_q
        return q_vec

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed media items using natural language."""
        if self.vectors is None or len(self.items) == 0:
            return []

        q_vec = self._embed_query(query)
        if np.linalg.norm(q_vec) < 1e-6:
            # Fallback to lexical search if query terms are entirely out of vocabulary
            results = []
            q_terms = set(self._tokenize(query))
            for item in self.items:
                text = f"{item.get('title', '')} {item.get('description', '')} {' '.join(item.get('tags', []))}".lower()
                matches = sum(1 for term in q_terms if term in text)
                if matches > 0:
                    results.append({"item": item, "score": float(matches) / len(q_terms)})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        scores = np.dot(self.vectors, q_vec)
        ranked_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "item": self.items[idx],
                "score": float(scores[idx])
            })
        return results

    @staticmethod
    def evaluate_motion_continuity(inter_frame_flows: list[float]) -> dict[str, Any]:
        """Evaluate a sequence of photos/frames for smooth motion reel generation."""
        if not inter_frame_flows:
            return {"eligible": False, "reason": "empty_sequence"}

        mean_flow = float(np.mean(inter_frame_flows))
        std_flow = float(np.std(inter_frame_flows))

        # Optical flow heuristic: mean between 0.05 and 15.0 px/frame indicates pleasant natural motion.
        # std_flow should be low relative to mean_flow (no wild erratic spikes).
        max_flow = float(np.max(inter_frame_flows))
        min_flow = float(np.min(inter_frame_flows))
        flow_range = max_flow - min_flow
        
        is_smooth = (
            0.05 <= mean_flow <= 15.0 and
            std_flow <= (mean_flow * 0.8 + 0.5) and
            flow_range <= (mean_flow * 2.5 + 1.0)
        )
        return {
            "eligible": is_smooth,
            "mean_flow": round(mean_flow, 4),
            "std_flow": round(std_flow, 4),
            "frame_count": len(inter_frame_flows) + 1,
            "recommendation": "suitable_for_reel" if is_smooth else "requires_motion_smoothing"
        }

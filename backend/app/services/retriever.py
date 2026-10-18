"""Local hybrid retrieval over Sol's prebuilt knowledge index."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")


class RetrieverConfigurationError(RuntimeError):
    """Raised when the saved RAG index is missing or inconsistent."""


def _backend_path(environment_name: str, default: str) -> Path:
    configured = Path(os.getenv(environment_name, default)).expanduser()
    return configured if configured.is_absolute() else BACKEND_DIR / configured


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def _keyword_score(question: str, record: dict[str, Any]) -> float:
    query_tokens = _tokens(question)
    if not query_tokens:
        return 0.0
    searchable = " ".join(
        str(record.get(field, ""))
        for field in ("title", "topic", "text", "organization", "source_id")
    )
    return len(query_tokens & _tokens(searchable)) / len(query_tokens)


def _priority_score(priority: Any) -> float:
    try:
        value = int(priority)
    except (TypeError, ValueError):
        return 0.0
    return {1: 1.0, 2: 0.6, 3: 0.3}.get(value, 0.0)


class Retriever:
    """Search normalized passage embeddings and rerank the best candidates."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or _backend_path("RAG_INDEX_PATH", "knowledge/index")
        self.config = self._load_config()
        self.records = self._load_metadata()
        self.embeddings = self._load_embeddings()
        self._validate_index()

        configured_model = os.getenv("EMBEDDING_MODEL", "").strip()
        indexed_model = str(self.config["embedding_model"])
        if configured_model and configured_model != indexed_model:
            raise RetrieverConfigurationError(
                "EMBEDDING_MODEL does not match the model used to build the index; "
                "rebuild the index before starting the backend"
            )

        # The index builder downloads the model. Runtime retrieval stays local
        # and does not unexpectedly make network requests.
        self.model = SentenceTransformer(indexed_model, local_files_only=True)
        self._encode_lock = Lock()

    def _load_config(self) -> dict[str, Any]:
        path = self.index_dir / "config.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RetrieverConfigurationError(
                f"RAG index is missing {path}; run scripts/build_rag_index.py"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RetrieverConfigurationError(f"Invalid RAG config: {exc}") from exc

    def _load_metadata(self) -> list[dict[str, Any]]:
        path = self.index_dir / "metadata.jsonl"
        try:
            with path.open(encoding="utf-8") as file:
                return [json.loads(line) for line in file if line.strip()]
        except FileNotFoundError as exc:
            raise RetrieverConfigurationError(f"RAG metadata is missing: {path}") from exc
        except json.JSONDecodeError as exc:
            raise RetrieverConfigurationError(f"Invalid RAG metadata: {exc}") from exc

    def _load_embeddings(self) -> np.ndarray:
        path = self.index_dir / "embeddings.npy"
        try:
            return np.load(path, allow_pickle=False)
        except FileNotFoundError as exc:
            raise RetrieverConfigurationError(f"RAG embeddings are missing: {path}") from exc

    def _validate_index(self) -> None:
        expected_count = int(self.config.get("record_count", -1))
        expected_dimensions = int(self.config.get("dimensions", -1))
        if self.embeddings.ndim != 2:
            raise RetrieverConfigurationError("RAG embeddings must be a two-dimensional matrix")
        if len(self.records) != expected_count or self.embeddings.shape[0] != expected_count:
            raise RetrieverConfigurationError("RAG embeddings and metadata counts do not match")
        if self.embeddings.shape[1] != expected_dimensions:
            raise RetrieverConfigurationError("RAG embedding dimensions do not match config")
        if not self.config.get("normalized"):
            raise RetrieverConfigurationError("RAG index must contain normalized embeddings")

    def search(
        self,
        question: str,
        *,
        limit: int = 6,
        candidate_limit: int = 20,
        minimum_semantic_score: float | None = None,
        minimum_final_score: float | None = None,
        max_chunks_per_source: int = 2,
    ) -> list[dict[str, Any]]:
        """Return diversified passages ordered by hybrid relevance."""
        question = question.strip()
        if not question:
            return []
        if limit < 1 or candidate_limit < limit:
            raise ValueError("candidate_limit must be greater than or equal to limit")

        if minimum_semantic_score is None:
            minimum_semantic_score = float(
                os.getenv("RAG_MIN_SEMANTIC_SCORE", "0.78")
            )
        if minimum_final_score is None:
            minimum_final_score = float(os.getenv("RAG_MIN_FINAL_SCORE", "0.78"))

        with self._encode_lock:
            query = self.model.encode(
                [f"query: {question}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].astype("float32")

        semantic_scores = self.embeddings @ query
        candidate_count = min(candidate_limit, len(self.records))
        candidate_indexes = np.argsort(semantic_scores)[::-1][:candidate_count]

        candidates: list[dict[str, Any]] = []
        for index in candidate_indexes:
            semantic = float(semantic_scores[index])
            if semantic < minimum_semantic_score:
                continue
            record = self.records[int(index)]
            keyword = _keyword_score(question, record)
            priority = _priority_score(record.get("priority"))
            final = 0.80 * semantic + 0.15 * keyword + 0.05 * priority
            if final < minimum_final_score:
                continue
            candidates.append(
                {
                    **record,
                    "semantic_score": semantic,
                    "keyword_score": keyword,
                    "priority_score": priority,
                    "final_score": final,
                }
            )

        candidates.sort(key=lambda item: item["final_score"], reverse=True)

        selected: list[dict[str, Any]] = []
        per_source: dict[str, int] = {}
        for candidate in candidates:
            source_id = str(candidate.get("source_id", "unknown"))
            if per_source.get(source_id, 0) >= max_chunks_per_source:
                continue
            selected.append(candidate)
            per_source[source_id] = per_source.get(source_id, 0) + 1
            if len(selected) >= limit:
                break
        return selected


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Create one retriever per backend process and reuse its loaded model."""
    return Retriever()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Search Sol's local RAG index")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--minimum-score", type=float, default=None)
    args = parser.parse_args()

    matches = get_retriever().search(
        args.question,
        limit=args.limit,
        minimum_semantic_score=args.minimum_score,
    )
    if not matches:
        print("No passage met the relevance threshold.")
        return
    for position, match in enumerate(matches, start=1):
        preview = " ".join(match["text"].split())[:180]
        print(
            f"{position}. [{match['source_id']}] score={match['final_score']:.3f} "
            f"semantic={match['semantic_score']:.3f} {match['title']}"
        )
        print(f"   {preview}")


if __name__ == "__main__":
    main()

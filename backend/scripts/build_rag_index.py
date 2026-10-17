"""Build the local semantic-search index used by Sol's RAG retriever."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def backend_path(environment_name: str, default: str) -> Path:
    configured = Path(os.getenv(environment_name, default)).expanduser()
    return configured if configured.is_absolute() else BACKEND_DIR / configured


CORPUS_PATH = backend_path(
    "RAG_KNOWLEDGE_BASE_PATH",
    "knowledge/corpus/knowledge_chunks.jsonl",
)
INDEX_DIR = backend_path("RAG_INDEX_PATH", "knowledge/index")
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
).strip()


def load_corpus(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            chunk_id = record.get("chunk_id")
            text = record.get("text")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Line {line_number} has no valid chunk_id")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Chunk {chunk_id} has no text")

            seen_ids.add(chunk_id)
            records.append(record)

    if not records:
        raise ValueError(f"No knowledge records found in {path}")
    return records


def passage_text(record: dict[str, Any]) -> str:
    title = record.get("title", "")
    topic = record.get("topic", "")
    return f"passage: {title}\nTopic: {topic}\n{record['text']}"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    records = load_corpus(CORPUS_PATH)
    corpus_hash = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [passage_text(record) for record in records],
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise RuntimeError("Embedding output does not match the knowledge corpus")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = INDEX_DIR / "embeddings.npy"
    temporary_embeddings = INDEX_DIR / "embeddings.tmp.npy"
    np.save(temporary_embeddings, embeddings)
    temporary_embeddings.replace(embeddings_path)

    write_jsonl(INDEX_DIR / "metadata.jsonl", records)

    config = {
        "embedding_model": MODEL_NAME,
        "record_count": len(records),
        "dimensions": int(embeddings.shape[1]),
        "normalized": True,
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": corpus_hash,
    }
    config_path = INDEX_DIR / "config.json"
    temporary_config = INDEX_DIR / "config.tmp.json"
    temporary_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_config.replace(config_path)

    print(
        f"Indexed {len(records)} chunks as {embeddings.shape[1]}-dimensional "
        f"vectors in {INDEX_DIR}"
    )


if __name__ == "__main__":
    main()

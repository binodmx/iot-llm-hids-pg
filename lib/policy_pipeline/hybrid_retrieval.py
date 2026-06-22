"""Lazy hybrid RAG retrieval: embed samples, retrieve top-k per class.

Provides concrete network flow examples to ground LLM reasoning, paired with
statistical aggregates. Used to improve generalization on imbalanced datasets.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def hybrid_retrieve(
    normal_train: pd.DataFrame,
    attack_train: pd.DataFrame,
    seed: int = 42,
    n_samples: int = 1000,
    n_retrieve: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Lazy hybrid retrieval: sample flows, embed, retrieve by centroid similarity.

    Args:
        normal_train: DataFrame of benign flows
        attack_train: DataFrame of attack flows
        seed: random seed for deterministic sampling
        n_samples: number of flows to sample per class before embedding
        n_retrieve: number of flows to retrieve per class for prompt

    Returns:
        (normal_examples, attack_examples): lists of dicts, top-k per class
    """

    # ─────────────────────────────────────────────────────────────────
    # Sample flows per class (lazy: don't embed full dataset)
    # ─────────────────────────────────────────────────────────────────

    sample_size_normal = min(n_samples, len(normal_train))
    sample_size_attack = min(n_samples, len(attack_train))

    sample_normal = normal_train.sample(
        n=sample_size_normal, random_state=seed, replace=False
    )
    sample_attack = attack_train.sample(
        n=sample_size_attack, random_state=seed, replace=False
    )

    # ─────────────────────────────────────────────────────────────────
    # Embed samples and build in-memory vector store
    # ─────────────────────────────────────────────────────────────────

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # lightweight, ~30MB
    )

    # Create in-memory Chroma store (no persistence)
    vector_store = Chroma(embedding_function=embeddings, collection_name="hybrid")

    # Convert rows to documents (JSON strings with metadata)
    normal_docs = [
        Document(
            page_content=json.dumps(row.to_dict()),
            metadata={"label": "normal", "idx": i},
        )
        for i, (_, row) in enumerate(sample_normal.iterrows())
    ]
    attack_docs = [
        Document(
            page_content=json.dumps(row.to_dict()),
            metadata={"label": "attack", "idx": i},
        )
        for i, (_, row) in enumerate(sample_attack.iterrows())
    ]

    # Add to vector store
    vector_store.add_documents(normal_docs)
    vector_store.add_documents(attack_docs)

    # ─────────────────────────────────────────────────────────────────
    # Compute class centroids and retrieve neighbors
    # ─────────────────────────────────────────────────────────────────

    # Embed all samples to compute centroid
    normal_embeddings = embeddings.embed_documents(
        [json.dumps(row.to_dict()) for _, row in sample_normal.iterrows()]
    )
    attack_embeddings = embeddings.embed_documents(
        [json.dumps(row.to_dict()) for _, row in sample_attack.iterrows()]
    )

    normal_centroid = np.mean(normal_embeddings, axis=0).tolist()
    attack_centroid = np.mean(attack_embeddings, axis=0).tolist()

    # Retrieve top-k by similarity to centroid
    normal_retrieved = vector_store.similarity_search_by_vector(
        normal_centroid, k=min(n_retrieve, len(normal_docs))
    )
    attack_retrieved = vector_store.similarity_search_by_vector(
        attack_centroid, k=min(n_retrieve, len(attack_docs))
    )

    # ─────────────────────────────────────────────────────────────────
    # Convert back to dicts (drop metadata, keep only flow data)
    # ─────────────────────────────────────────────────────────────────

    normal_examples = [json.loads(doc.page_content) for doc in normal_retrieved]
    attack_examples = [json.loads(doc.page_content) for doc in attack_retrieved]

    return normal_examples, attack_examples

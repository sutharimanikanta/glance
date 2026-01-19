# retriever\retrieve.py
"""
Two-stage retrieval with attribute-aware re-ranking.

Stage 1: Global CLIP similarity → top-N candidates (fast)
Stage 2: Attribute matching → re-rank candidates (precise)
"""

import numpy as np
import faiss
import pickle
from query_parser import QueryParser
from query_encoder import QueryEncoder


class FashionRetriever:
    """
    Retrieves fashion images using attribute-aware scoring.

    Scoring formula:
        final_score = 0.5 * global_sim
                    + 0.2 * color_sim
                    + 0.2 * clothing_sim
                    + 0.1 * environment_sim

    Weights can be tuned based on query type.
    """

    def __init__(self, index_dir="./index_data"):
        """Load FAISS index and metadata."""
        print("Loading index and metadata...")

        # Load FAISS index
        self.index = faiss.read_index(f"{index_dir}/faiss_index.bin")

        # Load metadata
        with open(f"{index_dir}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.image_filenames = metadata["image_filenames"]
            self.attribute_data = metadata["attribute_data"]
            attribute_prompts = metadata["attribute_prompts"]

        # Initialize query components
        self.parser = QueryParser(attribute_prompts)
        self.encoder = QueryEncoder(attribute_prompts)

        print(f"Loaded index with {len(self.image_filenames)} images")

    def _compute_attribute_similarity(self, query_scores, image_scores):
        """
        Compute similarity between query and image attribute scores.

        Args:
            query_scores: list of (index, score) from query
            image_scores: list of (index, score) from image

        Returns:
            float: similarity score [0, 1]
        """
        if not query_scores or not image_scores:
            return 0.0

        # Create score dictionaries
        query_dict = {idx: score for idx, score in query_scores}
        image_dict = {idx: score for idx, score in image_scores}

        # Compute overlap similarity
        common_indices = set(query_dict.keys()) & set(image_dict.keys())

        if not common_indices:
            return 0.0

        # Average of minimum scores for common attributes
        similarities = [min(query_dict[idx], image_dict[idx]) for idx in common_indices]
        return float(np.mean(similarities))

    def retrieve(self, query, top_k=10, candidate_pool=100, weights=None):
        """
        Retrieve top-k images for a query.

        Args:
            query: natural language query string
            top_k: number of results to return
            candidate_pool: number of candidates from stage 1
            weights: dict of scoring weights (optional)

        Returns:
            list of dicts: [
                {
                    'filename': image filename,
                    'score': final score,
                    'global_score': stage 1 score,
                    'breakdown': {color, clothing, environment, style scores}
                }
            ]
        """
        # Default weights
        if weights is None:
            weights = {"global": 0.5, "color": 0.2, "clothing": 0.2, "environment": 0.1}

        # Parse query
        parsed = self.parser.parse(query)
        print(f"\n{self.parser.format_parsed_query(parsed)}\n")

        # Encode query
        query_embeddings = self.encoder.encode_query(parsed)

        # Stage 1: Global retrieval
        query_vector = query_embeddings["global"].reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, candidate_pool)
        distances = distances[0]  # cosine similarities
        indices = indices[0]

        # Stage 2: Attribute re-ranking
        results = []

        for idx, global_score in zip(indices, distances):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            image_attrs = self.attribute_data[idx]

            # Compute attribute similarities
            color_sim = self._compute_attribute_similarity(
                query_embeddings["color_scores"], image_attrs["color"]
            )

            clothing_sim = self._compute_attribute_similarity(
                query_embeddings["clothing_scores"], image_attrs["clothing"]
            )

            environment_sim = self._compute_attribute_similarity(
                query_embeddings["environment_scores"], image_attrs["environment"]
            )

            style_sim = self._compute_attribute_similarity(
                query_embeddings["style_scores"], image_attrs["style"]
            )

            # Compute final score
            final_score = (
                weights["global"] * float(global_score)
                + weights["color"] * color_sim
                + weights["clothing"] * clothing_sim
                + weights["environment"] * environment_sim
            )

            results.append(
                {
                    "filename": self.image_filenames[idx],
                    "score": final_score,
                    "global_score": float(global_score),
                    "breakdown": {
                        "color": color_sim,
                        "clothing": clothing_sim,
                        "environment": environment_sim,
                        "style": style_sim,
                    },
                }
            )

        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def print_results(self, results):
        """Pretty print retrieval results."""
        print(f"Top {len(results)} Results:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']}")
            print(f"   Final Score: {result['score']:.4f}")
            print(
                f"   Global: {result['global_score']:.4f} | "
                f"Color: {result['breakdown']['color']:.4f} | "
                f"Clothing: {result['breakdown']['clothing']:.4f} | "
                f"Environment: {result['breakdown']['environment']:.4f}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve fashion images")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument(
        "--index_dir", type=str, default="./index_data", help="Index directory"
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of results")

    args = parser.parse_args()

    retriever = FashionRetriever(args.index_dir)
    results = retriever.retrieve(args.query, top_k=args.top_k)
    retriever.print_results(results)

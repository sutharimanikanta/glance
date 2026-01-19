# indexer\build_index.py
"""
Build FAISS index from fashion image dataset.

Creates:
1. FAISS index for global embeddings (fast retrieval)
2. Metadata storage for attribute embeddings (re-ranking)
"""

import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from image_encoder import MultiEmbeddingImageEncoder


def build_index(image_dir, output_dir="./index_data", batch_size=8):
    """
    Build FAISS index and metadata from image directory.

    Args:
        image_dir: directory containing fashion images
        output_dir: where to save index and metadata
        batch_size: images to process at once
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all image paths
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(list(Path(image_dir).glob(ext)))
    image_paths = [str(p) for p in image_paths]

    print(f"Found {len(image_paths)} images")

    # Initialize encoder
    encoder = MultiEmbeddingImageEncoder()

    # Storage for embeddings and metadata
    global_embeddings = []
    attribute_data = []
    image_filenames = []

    # Encode all images
    print("\nEncoding images...")
    for img_path, embeddings in tqdm(
        encoder.encode_images_batch(image_paths, batch_size), total=len(image_paths)
    ):
        global_embeddings.append(embeddings["global"])

        # Store attribute scores for re-ranking
        attribute_data.append(
            {
                "color": embeddings["color"],
                "clothing": embeddings["clothing"],
                "environment": embeddings["environment"],
                "style": embeddings["style"],
            }
        )

        image_filenames.append(os.path.basename(img_path))

    # Convert to numpy array
    global_embeddings = np.array(global_embeddings).astype("float32")
    print(f"\nGlobal embeddings shape: {global_embeddings.shape}")

    # Build FAISS index
    print("Building FAISS index...")
    dimension = global_embeddings.shape[1]

    # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)

    # Normalize for cosine similarity
    faiss.normalize_L2(global_embeddings)
    index.add(global_embeddings)

    print(f"Index built with {index.ntotal} vectors")

    # Save index and metadata
    print("\nSaving index and metadata...")
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))

    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(
            {
                "image_filenames": image_filenames,
                "attribute_data": attribute_data,
                "attribute_prompts": encoder.attribute_prompts,  # Save for query encoding
            },
            f,
        )

    print(f"\n✓ Index saved to {output_dir}")
    print(f"  - faiss_index.bin: {index.ntotal} vectors")
    print(f"  - metadata.pkl: {len(image_filenames)} images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index for fashion images")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./index_data", help="Output directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for encoding"
    )

    args = parser.parse_args()

    build_index(args.image_dir, args.output_dir, args.batch_size)

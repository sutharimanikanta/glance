# indexer\image_encoder.py
"""
Image encoder that creates multiple embeddings per image.

Key insight: Instead of one global embedding, we create:
1. Global image embedding (standard CLIP)
2. Attribute-specific embeddings by comparing image to attribute prompts

This enables compositional reasoning like "red shirt AND blue pants"
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from attribute_prompts import get_all_attribute_prompts


class MultiEmbeddingImageEncoder:
    """
    Encodes images into multiple attribute-aware embeddings.

    Architecture:
    - Global: full image → CLIP image encoder → 512D vector
    - Color: image vs color prompts → top-k similarity scores
    - Clothing: image vs clothing prompts → top-k similarity scores
    - Environment: image vs environment prompts → top-k similarity scores
    - Style: image vs style prompts → top-k similarity scores
    """

    def __init__(
        self, model_name="openai/clip-vit-base-patch32", device=None, top_k=10
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k  # Number of top attributes to keep per category

        print(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Pre-compute attribute prompt embeddings
        print("Pre-computing attribute prompt embeddings...")
        self.attribute_prompts = get_all_attribute_prompts()
        self.attribute_embeddings = self._encode_attribute_prompts()

    def _encode_attribute_prompts(self):
        """
        Pre-encode all attribute prompts to save computation during indexing.

        Returns:
            dict: {attribute_type: tensor of shape (num_prompts, embed_dim)}
        """
        encoded = {}

        with torch.no_grad():
            for attr_type, prompts in self.attribute_prompts.items():
                # Encode in batches to avoid memory issues
                batch_size = 32
                embeddings = []

                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i : i + batch_size]
                    inputs = self.processor(
                        text=batch, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    text_features = self.model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )
                    embeddings.append(text_features)

                encoded[attr_type] = torch.cat(embeddings, dim=0)
                print(f"  {attr_type}: {encoded[attr_type].shape[0]} prompts encoded")

        return encoded

    def encode_image(self, image_path):
        """
        Encode a single image into multi-embedding representation.

        Args:
            image_path: path to image file

        Returns:
            dict: {
                'global': 512D numpy array,
                'color': top-k (index, score) pairs,
                'clothing': top-k (index, score) pairs,
                'environment': top-k (index, score) pairs,
                'style': top-k (index, score) pairs
            }
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            # Global embedding
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            global_embed = image_features.cpu().numpy()[0]

            # Attribute embeddings: compute similarity with pre-encoded prompts
            attribute_scores = {}
            for attr_type, attr_embeds in self.attribute_embeddings.items():
                # Cosine similarity: image_features @ attr_embeds.T
                similarities = (image_features @ attr_embeds.T).squeeze(0)

                # Keep top-k attributes
                top_k_scores, top_k_indices = torch.topk(
                    similarities, k=min(self.top_k, len(similarities))
                )

                # Store as (index, score) pairs
                attribute_scores[attr_type] = list(
                    zip(top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy())
                )

        return {"global": global_embed, **attribute_scores}

    def encode_images_batch(self, image_paths, batch_size=8):
        """
        Encode multiple images efficiently.

        Args:
            image_paths: list of image paths
            batch_size: number of images to process at once

        Yields:
            tuple: (image_path, embeddings_dict)
        """
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            for path in batch_paths:
                try:
                    embeddings = self.encode_image(path)
                    yield path, embeddings
                except Exception as e:
                    print(f"Error encoding {path}: {e}")
                    continue

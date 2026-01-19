# retriever\query_encoder.py
"""
Encode parsed queries into embeddings for retrieval.

Creates:
1. Global query embedding from full text
2. Attribute-specific embeddings for detected attributes
"""

import torch
from transformers import CLIPProcessor, CLIPModel


class QueryEncoder:
    """
    Encodes text queries into multi-embedding representation.

    Matches the structure of image embeddings:
    - Global: full query → CLIP text encoder
    - Attributes: detected attributes → prompt embeddings
    """

    def __init__(
        self, attribute_prompts, model_name="openai/clip-vit-base-patch32", device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Pre-encode attribute prompts (same as in image encoder)
        self.attribute_prompts = attribute_prompts
        self.attribute_embeddings = self._encode_attribute_prompts()

    def _encode_attribute_prompts(self):
        """Pre-encode all attribute prompts."""
        encoded = {}

        with torch.no_grad():
            for attr_type, prompts in self.attribute_prompts.items():
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

        return encoded

    def encode_query(self, parsed_query):
        """
        Encode a parsed query into embeddings.

        Args:
            parsed_query: dict from QueryParser.parse()

        Returns:
            dict: {
                'global': 512D numpy array,
                'color_scores': list of (index, score) for detected colors,
                'clothing_scores': list of (index, score),
                'environment_scores': list of (index, score),
                'style_scores': list of (index, score)
            }
        """
        with torch.no_grad():
            # Global embedding from raw query
            inputs = self.processor(
                text=[parsed_query["raw_query"]], return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            global_embed = text_features.cpu().numpy()[0]

            # Attribute embeddings: find indices of detected attributes
            attribute_scores = {}

            # Map detected attributes to prompt indices and scores
            for attr_type in ["color", "clothing", "environment", "style"]:
                attr_key = f"{attr_type}s" if attr_type != "clothing" else "clothing"
                detected_attrs = parsed_query.get(attr_key, [])

                if detected_attrs:
                    # Find matching prompts and compute scores
                    scores = []
                    prompts = self.attribute_prompts[attr_type]

                    for attr in detected_attrs:
                        # Find prompts containing this attribute
                        matching_indices = [
                            i
                            for i, prompt in enumerate(prompts)
                            if attr in prompt.lower()
                        ]

                        # Compute similarity with matched prompts
                        if matching_indices:
                            matched_embeds = self.attribute_embeddings[attr_type][
                                matching_indices
                            ]

                            # Encode the attribute as a query
                            attr_inputs = self.processor(
                                text=[attr], return_tensors="pt", padding=True
                            )
                            attr_inputs = {
                                k: v.to(self.device) for k, v in attr_inputs.items()
                            }

                            attr_features = self.model.get_text_features(**attr_inputs)
                            attr_features = attr_features / attr_features.norm(
                                dim=-1, keepdim=True
                            )

                            # Compute similarities
                            sims = (attr_features @ matched_embeds.T).squeeze(0)

                            # Add to scores
                            for idx, sim in zip(matching_indices, sims.cpu().numpy()):
                                scores.append((idx, float(sim)))

                    # Sort by score and keep top ones
                    scores.sort(key=lambda x: x[1], reverse=True)
                    attribute_scores[f"{attr_type}_scores"] = scores[:10]
                else:
                    attribute_scores[f"{attr_type}_scores"] = []

        return {"global": global_embed, **attribute_scores}

# retriever\query_parser.py
"""
Parse natural language queries into structured attributes.

Uses simple rule-based matching + CLIP similarity to extract:
- Colors (red, blue, yellow)
- Clothing types (shirt, pants, dress)
- Environment (office, park, street)
- Style (formal, casual, sporty)

No LLMs needed - just keyword matching and CLIP embeddings.
"""

import re


class QueryParser:
    """
    Extracts fashion attributes from natural language queries.

    Strategy:
    1. Keyword matching for explicit mentions
    2. Context detection (e.g., "business" → formal style)
    """

    def __init__(self, attribute_prompts):
        """
        Args:
            attribute_prompts: dict from attribute_prompts.py
        """
        self.attribute_prompts = attribute_prompts

        # Build keyword sets for fast matching
        self.color_keywords = self._extract_keywords(attribute_prompts["color"])
        self.clothing_keywords = self._extract_keywords(attribute_prompts["clothing"])
        self.environment_keywords = self._extract_keywords(
            attribute_prompts["environment"]
        )
        self.style_keywords = self._extract_keywords(attribute_prompts["style"])

    def _extract_keywords(self, prompts):
        """Extract unique keywords from prompts."""
        keywords = set()
        for prompt in prompts:
            # Extract meaningful words (skip "a", "in", "wearing", etc.)
            words = re.findall(r"\b\w+\b", prompt.lower())
            keywords.update(
                [
                    w
                    for w in words
                    if len(w) > 2
                    and w
                    not in {"the", "and", "with", "wearing", "person", "photo", "taken"}
                ]
            )
        return keywords

    def parse(self, query):
        """
        Parse query into attribute components.

        Args:
            query: natural language string

        Returns:
            dict: {
                'raw_query': original query,
                'colors': list of detected colors,
                'clothing': list of detected clothing types,
                'environments': list of detected environments,
                'styles': list of detected styles
            }
        """
        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        # Match keywords
        colors = [w for w in words if w in self.color_keywords]
        clothing = [w for w in words if w in self.clothing_keywords]
        environments = [w for w in words if w in self.environment_keywords]
        styles = [w for w in words if w in self.style_keywords]

        # Context-based style inference
        if any(
            w in query_lower
            for w in ["business", "professional", "meeting", "office", "suit", "tie"]
        ):
            if "formal" not in styles:
                styles.append("formal")

        if any(
            w in query_lower for w in ["weekend", "relaxed", "everyday", "comfortable"]
        ):
            if "casual" not in styles:
                styles.append("casual")

        # Environment inference
        if "park" in query_lower or "outdoor" in query_lower or "bench" in query_lower:
            if "park" not in environments:
                environments.append("park")

        if "city" in query_lower or "urban" in query_lower or "street" in query_lower:
            if "city" not in environments:
                environments.append("city")

        return {
            "raw_query": query,
            "colors": colors,
            "clothing": clothing,
            "environments": environments,
            "styles": styles,
        }

    def format_parsed_query(self, parsed):
        """Pretty print parsed query for debugging."""
        lines = [f"Query: {parsed['raw_query']}"]
        if parsed["colors"]:
            lines.append(f"  Colors: {', '.join(parsed['colors'])}")
        if parsed["clothing"]:
            lines.append(f"  Clothing: {', '.join(parsed['clothing'])}")
        if parsed["environments"]:
            lines.append(f"  Environment: {', '.join(parsed['environments'])}")
        if parsed["styles"]:
            lines.append(f"  Style: {', '.join(parsed['styles'])}")
        return "\n".join(lines)

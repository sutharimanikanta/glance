# indexer\attribute_prompts.py
"""


Attribute vocabulary and prompt engineering for fashion retrieval.

Based on Fashionpedia structure, we define:
- Colors: common fashion colors
- Clothing types: shirts, pants, dresses, outerwear, etc.
- Environments: office, street, park, home, beach
- Styles: formal, casual, sporty, elegant

These are used to create targeted CLIP text embeddings.
"""

# Color vocabulary - common fashion colors
COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "brown",
    "beige",
    "navy",
    "maroon",
    "olive",
    "teal",
    "cream",
    "gold",
    "silver",
    "burgundy",
]

# Clothing type vocabulary
CLOTHING_TYPES = [
    "shirt",
    "t-shirt",
    "blouse",
    "sweater",
    "jacket",
    "coat",
    "pants",
    "jeans",
    "trousers",
    "shorts",
    "skirt",
    "dress",
    "suit",
    "blazer",
    "hoodie",
    "cardigan",
    "vest",
    "tie",
    "scarf",
    "hat",
    "shoes",
    "boots",
    "sneakers",
    "raincoat",
]

# Environment vocabulary
ENVIRONMENTS = [
    "office",
    "workplace",
    "business meeting",
    "street",
    "city",
    "urban area",
    "park",
    "outdoor",
    "garden",
    "home",
    "indoor",
    "living room",
    "beach",
    "seaside",
    "vacation",
    "cafe",
    "restaurant",
    "social setting",
]

# Style vocabulary
STYLES = [
    "formal",
    "business",
    "professional",
    "casual",
    "relaxed",
    "everyday",
    "sporty",
    "athletic",
    "active",
    "elegant",
    "sophisticated",
    "chic",
    "streetwear",
    "urban",
    "trendy",
]


def generate_color_prompts():
    """Generate color-focused prompts for CLIP encoding."""
    prompts = []
    for color in COLORS:
        prompts.extend(
            [
                f"a {color} shirt",
                f"a {color} dress",
                f"{color} clothing",
                f"wearing {color}",
            ]
        )
    return prompts


def generate_clothing_prompts():
    """Generate clothing-type prompts for CLIP encoding."""
    prompts = []
    for item in CLOTHING_TYPES:
        prompts.extend([f"a {item}", f"wearing a {item}", f"person in a {item}"])
    return prompts


def generate_environment_prompts():
    """Generate environment context prompts."""
    prompts = []
    for env in ENVIRONMENTS:
        prompts.extend([f"in {env}", f"at {env}", f"photo taken in {env}"])
    return prompts


def generate_style_prompts():
    """Generate style context prompts."""
    prompts = []
    for style in STYLES:
        prompts.extend([f"{style} outfit", f"{style} attire", f"{style} clothing"])
    return prompts


def get_all_attribute_prompts():
    """
    Get comprehensive attribute prompt dictionary.

    Returns:
        dict: {attribute_type: list of prompts}
    """
    return {
        "color": generate_color_prompts(),
        "clothing": generate_clothing_prompts(),
        "environment": generate_environment_prompts(),
        "style": generate_style_prompts(),
    }

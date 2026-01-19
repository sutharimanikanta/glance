# demo.py
"""
Complete demo script showing the full pipeline.

Usage:
    python demo.py --image_dir /path/to/images
"""

import os
import sys
import argparse

# Add paths
sys.path.append("./indexer")
sys.path.append("./retriever")


def demo_full_pipeline(image_dir, output_dir="./index_data"):
    """
    Demonstrate the complete fashion retrieval pipeline.

    1. Build index
    2. Run example queries
    3. Show attribute analysis
    """

    print("=" * 80)
    print("FASHION RETRIEVAL SYSTEM - FULL DEMO")
    print("=" * 80)

    # ========================================================================
    # STEP 1: BUILD INDEX
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: BUILDING INDEX")
    print("=" * 80)

    from indexer.build_index import build_index

    build_index(image_dir=image_dir, output_dir=output_dir, batch_size=8)

    # ========================================================================
    # STEP 2: INITIALIZE RETRIEVER
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING RETRIEVER")
    print("=" * 80)

    from retriever.retrieve import FashionRetriever

    retriever = FashionRetriever(index_dir=output_dir)

    # ========================================================================
    # STEP 3: EXAMPLE QUERIES
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: RUNNING EXAMPLE QUERIES")
    print("=" * 80)

    example_queries = [
        {
            "query": "A person in a bright yellow raincoat",
            "description": "Single color + specific clothing item",
        },
        {
            "query": "Professional business attire inside a modern office",
            "description": "Style + environment context",
        },
        {
            "query": "Someone wearing a blue shirt sitting on a park bench",
            "description": "Color + clothing + environment + action",
        },
        {
            "query": "red shirt with blue pants",
            "description": "Multiple color attributes (compositional)",
        },
    ]

    for i, example in enumerate(example_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"EXAMPLE {i}: {example['description']}")
        print(f"{'=' * 80}")

        results = retriever.retrieve(example["query"], top_k=5)
        retriever.print_results(results)

    # ========================================================================
    # STEP 4: WEIGHT COMPARISON
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 4: COMPARING DIFFERENT WEIGHT CONFIGURATIONS")
    print("=" * 80)

    query = "red dress"

    weight_configs = [
        {
            "name": "Balanced (Default)",
            "weights": {
                "global": 0.5,
                "color": 0.2,
                "clothing": 0.2,
                "environment": 0.1,
            },
        },
        {
            "name": "Color-Focused",
            "weights": {
                "global": 0.2,
                "color": 0.6,
                "clothing": 0.15,
                "environment": 0.05,
            },
        },
        {
            "name": "Global-Heavy (Like Vanilla CLIP)",
            "weights": {
                "global": 0.9,
                "color": 0.05,
                "clothing": 0.03,
                "environment": 0.02,
            },
        },
    ]

    print(f"\nQuery: '{query}'")

    for config in weight_configs:
        print(f"\n{'-' * 80}")
        print(f"Configuration: {config['name']}")
        print(f"Weights: {config['weights']}")
        print(f"{'-' * 80}")

        results = retriever.retrieve(query, top_k=3, weights=config["weights"])

        for j, result in enumerate(results, 1):
            print(f"\n  {j}. {result['filename']}")
            print(
                f"     Final: {result['score']:.4f} | "
                f"Global: {result['global_score']:.4f} | "
                f"Color: {result['breakdown']['color']:.4f}"
            )

    # ========================================================================
    # STEP 5: QUERY ANALYSIS
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 5: ANALYZING QUERY PARSING")
    print("=" * 80)

    analysis_queries = [
        "red shirt with blue pants",
        "formal business meeting attire",
        "casual weekend city walk",
        "elegant black dress at restaurant",
    ]

    for query in analysis_queries:
        parsed = retriever.parser.parse(query)
        print(f"\n{retriever.parser.format_parsed_query(parsed)}")

    # ========================================================================
    # STEP 6: STATISTICS
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 6: SYSTEM STATISTICS")
    print("=" * 80)

    print("\nIndex Statistics:")
    print(f"  Total images indexed: {len(retriever.image_filenames)}")
    print(f"  FAISS index size: {retriever.index.ntotal} vectors")

    print("\nAttribute Vocabulary:")
    for attr_type, prompts in retriever.parser.attribute_prompts.items():
        print(f"  {attr_type}: {len(prompts)} prompts")

    print("\nMemory Usage (approximate):")
    global_size = retriever.index.ntotal * 512 * 4 / (1024**2)  # MB
    print(f"  Global embeddings: ~{global_size:.1f} MB")
    print(f"  Metadata: ~{len(retriever.attribute_data) * 2 / 1024:.1f} MB")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)

    print("\n✓ Index built successfully")
    print("✓ Retrieval system working")
    print("✓ Attribute-aware scoring demonstrated")

    print("\nNext steps:")
    print(
        "  1. Try your own queries: python retriever/retrieve.py --query 'your query'"
    )
    print("  2. Start Flask API: python app.py")
    print("  3. Run full tests: python test_queries.py")

    print("\nSee README.md for detailed documentation.")


def quick_search_demo(index_dir="./index_data"):
    """Quick demo if index already exists."""

    from retriever.retrieve import FashionRetriever

    print("=" * 80)
    print("QUICK SEARCH DEMO")
    print("=" * 80)

    retriever = FashionRetriever(index_dir=index_dir)

    print("\nTry these commands:")
    print("  1. Single attribute: 'red dress'")
    print("  2. Compositional: 'red shirt with blue pants'")
    print("  3. Context: 'formal outfit in office'")
    print("  4. Complex: 'casual weekend city outfit with sneakers'")

    print("\nEnter your query (or 'quit' to exit):")

    while True:
        query = input("\n> ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            continue

        try:
            results = retriever.retrieve(query, top_k=5)
            retriever.print_results(results)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fashion Retrieval System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline demo (build index + search)
  python demo.py --image_dir /path/to/images

  # Quick search demo (if index exists)
  python demo.py --quick
        """,
    )

    parser.add_argument(
        "--image_dir", type=str, help="Directory containing fashion images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./index_data",
        help="Output directory for index",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick demo mode (index must already exist)",
    )

    args = parser.parse_args()

    if args.quick:
        # Quick search demo
        quick_search_demo(args.output_dir)
    else:
        # Full pipeline demo
        if not args.image_dir:
            parser.error("--image_dir is required for full demo")

        if not os.path.exists(args.image_dir):
            parser.error(f"Image directory not found: {args.image_dir}")

        demo_full_pipeline(args.image_dir, args.output_dir)

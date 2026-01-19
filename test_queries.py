# test_queries.py
"""
Test script with example queries from the assignment.

Run this after building the index to verify the system works.
"""

import sys

sys.path.append("./retriever")

from retriever.retrieve import FashionRetriever


def test_all_queries():
    """Test all required evaluation queries."""

    # Initialize retriever
    print("Initializing retriever...\n")
    retriever = FashionRetriever(index_dir="./index_data")

    # Test queries from assignment
    test_queries = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting",
    ]

    print("=" * 80)
    print("TESTING FASHION RETRIEVAL SYSTEM")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'=' * 80}")

        # Retrieve
        results = retriever.retrieve(query, top_k=5)

        # Print results
        retriever.print_results(results)

        print("\n")


def test_custom_weights():
    """Test with custom attribute weights."""

    retriever = FashionRetriever(index_dir="./index_data")

    query = "red shirt with blue pants"

    print("\n" + "=" * 80)
    print("TESTING CUSTOM WEIGHTS")
    print("=" * 80)

    # Emphasize color matching
    print("\n1. Heavy color weighting (color is most important):")
    results = retriever.retrieve(
        query,
        top_k=5,
        weights={
            "global": 0.2,
            "color": 0.6,  # Boost color importance
            "clothing": 0.1,
            "environment": 0.1,
        },
    )
    retriever.print_results(results)

    # Emphasize global similarity
    print("\n2. Heavy global weighting (more like vanilla CLIP):")
    results = retriever.retrieve(
        query,
        top_k=5,
        weights={
            "global": 0.9,  # Almost vanilla CLIP
            "color": 0.05,
            "clothing": 0.03,
            "environment": 0.02,
        },
    )
    retriever.print_results(results)


def analyze_query_parsing():
    """Analyze how queries are parsed."""

    retriever = FashionRetriever(index_dir="./index_data")

    test_queries = [
        "red shirt with blue pants",
        "formal business attire in office",
        "casual weekend city walk",
        "yellow raincoat in rainy weather",
        "elegant dress at a restaurant",
    ]

    print("\n" + "=" * 80)
    print("QUERY PARSING ANALYSIS")
    print("=" * 80)

    for query in test_queries:
        parsed = retriever.parser.parse(query)
        print(f"\n{retriever.parser.format_parsed_query(parsed)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test fashion retrieval system")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "weights", "parsing"],
        help="Test mode: all queries, custom weights, or query parsing",
    )

    args = parser.parse_args()

    if args.mode == "all":
        test_all_queries()
    elif args.mode == "weights":
        test_custom_weights()
    elif args.mode == "parsing":
        analyze_query_parsing()

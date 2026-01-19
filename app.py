# app.py
"""
Simple Flask API for fashion image retrieval.

Endpoints:
- POST /search: search for images
- GET /health: health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

sys.path.append("./retriever")

from retriever.retrieve import FashionRetriever

app = Flask(__name__)
CORS(app)

# Initialize retriever (load once at startup)
print("Initializing retriever...")
retriever = FashionRetriever(index_dir="./index_data")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "num_images": len(retriever.image_filenames)})


@app.route("/search", methods=["POST"])
def search():
    """
    Search for fashion images.

    Request body:
    {
        "query": "red shirt with blue pants",
        "top_k": 10,
        "weights": {  // optional
            "global": 0.5,
            "color": 0.2,
            "clothing": 0.2,
            "environment": 0.1
        }
    }

    Response:
    {
        "query": "...",
        "results": [
            {
                "filename": "image1.jpg",
                "score": 0.85,
                "global_score": 0.78,
                "breakdown": {
                    "color": 0.92,
                    "clothing": 0.85,
                    "environment": 0.45,
                    "style": 0.60
                }
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        query = data.get("query")
        top_k = data.get("top_k", 10)
        weights = data.get("weights", None)

        if not query:
            return jsonify({"error": "query is required"}), 400

        # Perform retrieval
        results = retriever.retrieve(query=query, top_k=top_k, weights=weights)

        return jsonify(
            {"query": query, "num_results": len(results), "results": results}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

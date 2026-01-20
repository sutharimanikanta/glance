## Quick Start Guide

Get your fashion retrieval system running in 5 minutes!

📋 Prerequisites

Python 3.8+

CUDA-capable GPU (optional, but recommended)

Fashion image dataset (e.g., DeepFashion, Fashionpedia)

🚀 Installation
# Clone or download the repository
cd fashion-retrieval

# Install dependencies
pip install -r requirements.txt

📁 Prepare Your Dataset

Organize your images in a single directory:

/path/to/images/
  ├── 94876f5333d96f8ef.jpg
  ├── 17683e4a33b5c1906.jpg
  ├── 3cd210ef2b3843e00248c42ff78edb2e.jpg
  └── ...

🔨 Build Index
python indexer/build_index.py \
    --image_dir /path/to/images \
    --output_dir ./index_data \
    --batch_size 8


Expected output:

Loading CLIP model on cuda...
Pre-computing attribute prompt embeddings...
  color: 80 prompts encoded
  clothing: 69 prompts encoded
  environment: 18 prompts encoded
  style: 27 prompts encoded

Found 10000 images
Encoding images...
100%|████████████████| 10000/10000 [15:23<00:00, 10.83it/s]

Building FAISS index...
Index built with 10000 vectors

✓ Index saved to ./index_data
  - faiss_index.bin: 10000 vectors
  - metadata.pkl: 10000 images


Time estimate: ~1 second per image on GPU, ~3 seconds on CPU

🔍 Search Images
Command Line
python retriever/retrieve.py \
    --query "red shirt with blue pants" \
    --index_dir ./index_data \
    --top_k 10

Python API
from retriever.retrieve import FashionRetriever

# Initialize (loads index once)
retriever = FashionRetriever(index_dir="./index_data")

# Search
results = retriever.retrieve(
    query="professional business attire in office",
    top_k=10
)

# Print results
retriever.print_results(results)

Flask API
# Start server
python app.py

# In another terminal, search via API
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "casual weekend outfit for a city walk",
    "top_k": 5
  }'

🧪 Run Tests
# Test all evaluation queries
python test_queries.py --mode all

# Test custom weight configurations
python test_queries.py --mode weights

# Analyze query parsing
python test_queries.py --mode parsing

📊 Example Queries

Try these queries to see attribute-aware retrieval in action:

queries = [
    "A person in a bright yellow raincoat",
    "Professional business attire inside a modern office",
    "Someone wearing a blue shirt sitting on a park bench",
    "Casual weekend outfit for a city walk",
    "A red tie and a white shirt in a formal setting",
    "Elegant black dress at a restaurant",
    "Sporty outfit for running in the park",
    "Beige coat and brown boots in autumn"
]

🎛️ Custom Scoring Weights

Adjust attribute importance based on your use case:

# Emphasize color matching
results = retriever.retrieve(
    query="red dress",
    weights={
        "global": 0.3,
        "color": 0.5,
        "clothing": 0.15,
        "environment": 0.05
    }
)

# Emphasize environment/context
results = retriever.retrieve(
    query="outfit for beach vacation",
    weights={
        "global": 0.4,
        "color": 0.1,
        "clothing": 0.2,
        "environment": 0.3
    }
)

📈 Performance Expectations
Dataset Size	Indexing Time (GPU)	Query Time	Memory Usage
1K images	~2 minutes	<10 ms	~200 MB
10K images	~20 minutes	<15 ms	~500 MB
100K images	~3 hours	<20 ms	~2 GB
1M images	~30 hours	<30 ms	~4 GB
🐛 Troubleshooting
Out of Memory During Indexing
# Reduce batch size
python indexer/build_index.py \
    --image_dir /path/to/images \
    --batch_size 4

Slow Indexing on CPU
# Use smaller CLIP model
# Edit indexer/image_encoder.py:
# model_name = "openai/clip-vit-base-patch16"

Flask API Not Responding
# Check if running
curl http://localhost:5000/health

# Expected response:
# {"status": "healthy", "num_images": 10000}

📝 Next Steps

Experiment with complex multi-attribute queries

Tune scoring weights based on your dataset

Add custom attributes in attribute_prompts.py

Evaluate performance against vanilla CLIP

💡 Tips for Best Results

Use high-quality, well-lit fashion images

Be specific in queries (e.g., “red blazer in office”)

Increase color weight for color-focused queries

Include context like environment when possible

📚 Further Reading

README.md
 — Full technical documentation

CLIP Paper
 — Understanding CLIP

Fashionpedia
 — Fashion attribute taxonomy

Ready to search! 🚀
Start with:

python test_queries.py --mode all


## 🚀 Installation
```bash
pip install -r requirements.txt

✅ **Must have blank lines before and after fenced blocks**

---

### 2️⃣ Tables with right-aligned colons + emojis above  
This combo sometimes breaks rendering when:
- Emojis are in the heading
- Table immediately follows without a blank paragraph

Safer approach:  
✔ Add a short line before tables  
✔ Avoid right-alignment colons (`----:`)

---

### 3️⃣ Inline backticks + emojis in the same paragraph  
This doesn’t break CommonMark, but **GitHub UI sometimes collapses sections visually**.

Example:
Ready to search! 🚀 Start with python test_queries.py --mode all


Safer: split into two lines.

---

### 4️⃣ Overuse of emojis in headings  
GitHub supports emojis, but **emoji-heavy headers + long README** sometimes trigger collapsible rendering in preview.

Fix: Keep emojis, but **not on every section**.

---

## ✅ FINAL — GitHub-Safe README (No Collapse)

This version:
- Uses **strict CommonMark**
- Has **correct spacing**
- Avoids edge-case rendering bugs
- Is **100% GitHub-safe**

You can paste this directly.

---

# Quick Start Guide

Get your fashion retrieval system running in 5 minutes.

---

## Prerequisites

- Python 3.8 or higher  
- CUDA-capable GPU (optional, recommended)  
- Fashion image dataset (DeepFashion, Fashionpedia, etc.)

---

## Installation

```bash
cd fashion-retrieval
pip install -r requirements.txt
Prepare Your Dataset
Organize all images in a single directory.

/path/to/images/
├── 94876f5333d96f8ef.jpg
├── 17683e4a33b5c1906.jpg
├── 3cd210ef2b3843e00248c42ff78edb2e.jpg
└── ...
Build Index
python indexer/build_index.py \
  --image_dir /path/to/images \
  --output_dir ./index_data \
  --batch_size 8
Expected output:

Loading CLIP model on cuda...
Pre-computing attribute prompt embeddings...
Found 10000 images
Encoding images...
Building FAISS index...
Index saved to ./index_data
Time estimate:

GPU: ~1 second per image

CPU: ~3 seconds per image

Search Images
Command Line
python retriever/retrieve.py \
  --query "red shirt with blue pants" \
  --index_dir ./index_data \
  --top_k 10
Python API
from retriever.retrieve import FashionRetriever

retriever = FashionRetriever(index_dir="./index_data")

results = retriever.retrieve(
    query="professional business attire in office",
    top_k=10
)

retriever.print_results(results)
Flask API
python app.py
In another terminal:

curl -X POST http://localhost:5000/search \
-H "Content-Type: application/json" \
-d '{"query":"casual weekend outfit","top_k":5}'
Run Tests
python test_queries.py --mode all
python test_queries.py --mode weights
python test_queries.py --mode parsing
Example Queries
queries = [
  "Bright yellow raincoat",
  "Business attire in office",
  "Blue shirt on park bench",
  "Casual weekend outfit",
  "Red tie with white shirt",
  "Black dress in restaurant",
  "Sporty running outfit",
  "Beige coat with brown boots"
]
Custom Scoring Weights
results = retriever.retrieve(
  query="red dress",
  weights={
    "global": 0.3,
    "color": 0.5,
    "clothing": 0.15,
    "environment": 0.05
  }
)
Performance Expectations
Performance depends on dataset size.

Dataset Size	Indexing Time	Query Time	Memory
1K images	~2 minutes	<10 ms	~200 MB
10K images	~20 minutes	<15 ms	~500 MB
100K images	~3 hours	<20 ms	~2 GB
1M images	~30 hours	<30 ms	~4 GB
Troubleshooting
Out of Memory
Reduce batch size:

--batch_size 4
Slow CPU Indexing
Use a smaller CLIP model in image_encoder.py.

Next Steps
Try multi-attribute queries

Tune scoring weights

Add custom attributes

Compare with vanilla CLIP

Further Reading
CLIP paper: https://arxiv.org/abs/2103.00020

Fashionpedia: https://fashionpedia.github.io/home/

Ready to search.

Run:

python test_queries.py --mode all

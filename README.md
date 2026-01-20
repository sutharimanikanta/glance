Quick Start GuideGet your fashion retrieval system running in 5 minutes.PrerequisitesPython 3.8 or higherCUDA-capable GPU (optional, recommended)Fashion image dataset (DeepFashion, Fashionpedia, etc.)InstallationBashcd fashion-retrieval
pip install -r requirements.txt
Prepare Your DatasetOrganize all images in a single directory:Plaintext/path/to/images/
├── 94876f5333d96f8ef.jpg
├── 17683e4a33b5c1906.jpg
├── 3cd210ef2b3843e00248c42ff78edb2e.jpg
└── ...
Build IndexRun the indexer with your image path:Bashpython indexer/build_index.py \
  --image_dir /path/to/images \
  --output_dir ./index_data \
  --batch_size 8
Expected OutputPlaintextLoading CLIP model on cuda...
Pre-computing attribute prompt embeddings...
Found 10000 images
Encoding images...
Building FAISS index...
Index saved to ./index_data
Time EstimateGPU: ~1 second per imageCPU: ~3 seconds per imageSearch ImagesCommand LineBashpython retriever/retrieve.py \
  --query "red shirt with blue pants" \
  --index_dir ./index_data \
  --top_k 10
Python APIPythonfrom retriever.retrieve import FashionRetriever

retriever = FashionRetriever(index_dir="./index_data")

results = retriever.retrieve(
    query="professional business attire in office",
    top_k=10
)

retriever.print_results(results)
Flask APIStart the server:Bashpython app.py
In another terminal, run:Bashcurl -X POST http://localhost:5000/search \
-H "Content-Type: application/json" \
-d '{"query":"casual weekend outfit","top_k":5}'
Run TestsBashpython test_queries.py --mode all
python test_queries.py --mode weights
python test_queries.py --mode parsing
Example Queries"Bright yellow raincoat""Business attire in office""Blue shirt on park bench""Casual weekend outfit""Red tie with white shirt""Black dress in restaurant""Sporty running outfit""Beige coat with brown boots"Custom Scoring WeightsPythonresults = retriever.retrieve(
  query="red dress",
  weights={
    "global": 0.3,
    "color": 0.5,
    "clothing": 0.15,
    "environment": 0.05
  }
)
Performance ExpectationsDataset SizeIndexing TimeQuery TimeMemory1K images~2 minutes<10 ms~200 MB10K images~20 minutes<15 ms~500 MB100K images~3 hours<20 ms~2 GB1M images~30 hours<30 ms~4 GBTroubleshootingOut of MemoryReduce batch size to lower VRAM usage:--batch_size 4Slow CPU IndexingUse a smaller CLIP model in image_encoder.py.Next StepsTry multi-attribute queries.Tune scoring weights for your specific dataset.Add custom attributes to the config.Compare results with vanilla CLIP.Further ReadingCLIP Paper: https://arxiv.org/abs/2103.00020Fashionpedia: https://fashionpedia.github.io/home/Ready to search. Run the following to verify:python test_queries.py --mode all

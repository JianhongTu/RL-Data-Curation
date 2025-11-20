import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path

def build_alpaca_embeddings(batch_size: int=512, device: str | None = None, cache_dir: str = "./data"):
  """
  Build embeddings for the Alpaca dataset using all-mpnet-base-v2.
  Returns normalized embeddings (unit vectors) for use with cosine distance.
  
  Embeddings are cached to disk to avoid recomputing on subsequent runs.
  """
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  # Setup cache directory and file
  cache_path = Path(cache_dir)
  cache_path.mkdir(exist_ok=True, parents=True)
  embeddings_file = cache_path / "alpaca_embeddings_mpnet.npz"
  
  # Try to load from cache
  if embeddings_file.exists():
    print(f"Loading cached embeddings from {embeddings_file}...")
    cached = np.load(embeddings_file)
    X_unit_alpaca = cached['embeddings']
    y = cached['labels']
    print(f"✓ Loaded {len(X_unit_alpaca)} cached embeddings (dim={X_unit_alpaca.shape[1]})")
    return X_unit_alpaca, y
  
  # Cache miss - compute embeddings
  print("Cache miss - computing embeddings from scratch...")
  print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
  mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
  
  print("Loading Alpaca dataset...")
  alpaca_ds = load_dataset("yahma/alpaca-cleaned")
  train_ds = alpaca_ds["train"]

  print(f"Processing {len(train_ds)} examples...")
  texts = []
  for ex in train_ds:
    instr = ex["instruction"].strip()
    inp = ex["input"].strip()
    if inp:
      txt = instr + " " + inp
    else:
      txt = instr
    texts.append(txt)

  print("Encoding texts with all-mpnet-base-v2 (this may take a few minutes)...")
  X_embed_alpaca = mpnet.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
  
  print("Normalizing embeddings...")
  norms = np.linalg.norm(X_embed_alpaca, axis=1, keepdims=True) + 1e-8
  X_unit_alpaca = (X_embed_alpaca / norms).astype(np.float32)

  # Return both embeddings and dummy labels (for compatibility)
  y = np.zeros(len(X_unit_alpaca), dtype=np.int64)
  
  # Save to cache
  print(f"Saving embeddings to cache: {embeddings_file}")
  np.savez_compressed(embeddings_file, embeddings=X_unit_alpaca, labels=y)
  print("✓ Embeddings cached for future runs")
  
  return X_unit_alpaca, y
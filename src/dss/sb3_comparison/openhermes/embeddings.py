import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path

def build_openhermes_embeddings(batch_size: int=512, device: str | None = None, cache_dir: str = "./data"):
  """
  Build embeddings for the OpenHermes 2.5 dataset using all-mpnet-base-v2.
  Returns normalized embeddings (unit vectors) for use with cosine distance.
  
  OpenHermes 2.5 is designed to align with benchmarks like MMLU, ARC Challenge,
  BoolQ, and HellaSwag, making it a better choice than Alpaca for these evaluations.
  
  Embeddings are cached to disk to avoid recomputing on subsequent runs.
  """
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  # Setup cache directory and file
  cache_path = Path(cache_dir)
  cache_path.mkdir(exist_ok=True, parents=True)
  embeddings_file = cache_path / "openhermes_embeddings_mpnet.npz"
  
  # Try to load from cache
  if embeddings_file.exists():
    print(f"Loading cached embeddings from {embeddings_file}...")
    cached = np.load(embeddings_file)
    X_unit_openhermes = cached['embeddings']
    y = cached['labels']
    print(f"✓ Loaded {len(X_unit_openhermes)} cached embeddings (dim={X_unit_openhermes.shape[1]})")
    return X_unit_openhermes, y
  
  # Cache miss - compute embeddings
  print("Cache miss - computing embeddings from scratch...")
  print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
  mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

  print("Loading OpenHermes 2.5 dataset...")
  # OpenHermes 2.5 dataset on HuggingFace
  try:
    openhermes_ds = load_dataset("teknium/OpenHermes-2.5")
  except Exception as e:
    print(f"Error loading 'teknium/OpenHermes-2.5', trying alternative...")
    try:
      openhermes_ds = load_dataset("teknium/openhermes", "default")
    except Exception as e2:
      print(f"Error loading alternative, trying 'WizardLM/WizardLM_evol_instruct_V2_196k'...")
      openhermes_ds = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k")
  
  # Handle different dataset splits
  if "train" in openhermes_ds:
    train_ds = openhermes_ds["train"]
  elif len(openhermes_ds) > 0:
    # If no split, use the first available split
    train_ds = list(openhermes_ds.values())[0]
  else:
    raise ValueError("Could not find train split in OpenHermes dataset")

  print(f"Processing {len(train_ds)} examples...")
  texts = []  # Will store extracted text strings, one per example
  
  # OpenHermes uses conversational format with "conversations" field
  # Each conversation entry has "from" (role: "human", "gpt", etc.) and "value" (content)
  # Example: {"conversations": [{"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "4"}]}
  for ex in train_ds:
    if "conversations" not in ex:
      raise ValueError(f"Expected 'conversations' field in OpenHermes example. Found keys: {list(ex.keys())}")
    
    # Combine all conversation turns into one text string with role prefixes
    conv_texts = []
    for conv in ex["conversations"]:
      role = conv.get("from", "")  # "human", "gpt", "system", etc.
      content = conv.get("value", "").strip()  # The actual message text
      if content:
        # Map common roles to readable prefixes
        if role == "human":
          conv_texts.append(f"Human: {content}")
        elif role == "gpt":
          conv_texts.append(f"Assistant: {content}")
        elif role == "system":
          conv_texts.append(f"System: {content}")
        else:
          # Unknown role, use it as-is or with generic prefix
          conv_texts.append(f"{role.capitalize()}: {content}" if role else content)
    
    txt = " ".join(conv_texts)
    if txt.strip():
      texts.append(txt)

  if not texts:
    raise ValueError("No text examples found in OpenHermes dataset. Check dataset format.")

  # ========================================================================
  # EMBEDDING GENERATION: Convert text strings to 768-dim vectors
  # ========================================================================
  print(f"Extracted {len(texts)} text examples for embedding...")
  print("Encoding texts with all-mpnet-base-v2 (this may take a few minutes)...")
  
  # Encode all texts to embeddings
  # Input:  List of text strings (e.g., ["User: What is 2+2? ...", ...])
  # Output: NumPy array of shape (num_texts, 768)
  # Process: Tokenize → MPNet transformer → Pool → 768-dim vector per text
  # Note: all-mpnet-base-v2 has max_seq_length=384 tokens, longer texts are truncated
  X_embed_openhermes = mpnet.encode(
      texts,                    # List of text strings
      batch_size=64,            # Process 64 texts at a time (memory efficient)
      show_progress_bar=True,   # Show progress bar
      convert_to_numpy=True,   # Return as NumPy array (not PyTorch tensor)
      normalize_embeddings=False  # We'll normalize manually after
  ).astype(np.float32)          # Convert to float32 for memory efficiency
  
  # ========================================================================
  # NORMALIZATION: Convert to unit vectors (length = 1.0) for cosine distance
  # ========================================================================
  # Why normalize?
  # - Cosine distance = 1 - cosine_similarity
  # - Cosine similarity = dot product of normalized vectors
  # - Normalized vectors make cosine distance calculations efficient
  print("Normalizing embeddings...")
  
  # Compute L2 norm (Euclidean length) for each embedding
  # norms.shape = (num_examples, 1)
  # norms[i] = sqrt(sum(X_embed_openhermes[i]^2))
  norms = np.linalg.norm(X_embed_openhermes, axis=1, keepdims=True) + 1e-8
  
  # Divide each embedding by its norm to get unit vectors
  # Result: Each embedding has length ≈ 1.0
  X_unit_openhermes = (X_embed_openhermes / norms).astype(np.float32)

  # Return both embeddings and dummy labels (for compatibility)
  y = np.zeros(len(X_unit_openhermes), dtype=np.int64)
  
  # Save to cache
  print(f"Saving embeddings to cache: {embeddings_file}")
  np.savez_compressed(embeddings_file, embeddings=X_unit_openhermes, labels=y)
  print("✓ Embeddings cached for future runs")
  
  return X_unit_openhermes, y


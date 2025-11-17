import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def build_alpaca_embeddings(batch_size: int = 64, device: str | None = None):
    """
    Build Alpaca embeddings using sentence-transformers/all-mpnet-base-v2.
    Loads data from HuggingFace datasets (yahma/alpaca-cleaned).
    
    Args:
        batch_size: Batch size for encoding
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
    
    Returns:
        X_embed: Embedding matrix of shape (N, 768)
        y: Dummy labels (all zeros) for compatibility
    """
    if device is None:
        device = "cuda" if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available() else "cpu"
    
    # Load Alpaca dataset from HuggingFace
    print("Loading Alpaca dataset from HuggingFace (yahma/alpaca-cleaned)...")
    alpaca_ds = load_dataset("yahma/alpaca-cleaned")
    train_ds = alpaca_ds["train"]
    
    # Extract instruction texts (combine instruction, input, and output)
    texts = []
    for ex in train_ds:
        instr = ex["instruction"].strip()
        inp = ex["input"].strip()
        if inp:
            text = instr + " " + inp
        else:
            text = instr
        texts.append(text)
    
    print(f"Loaded {len(texts)} Alpaca examples")
    
    # Load sentence transformer model
    print("Loading sentence transformer model (all-mpnet-base-v2)...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    
    # Encode texts to embeddings
    print(f"Encoding texts to embeddings (batch_size={batch_size})...")
    X_embed = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    
    print(f"Generated embeddings shape: {X_embed.shape}")
    
    # Return dummy labels for compatibility
    y = np.zeros(len(X_embed), dtype=np.int64)
    
    return X_embed, y

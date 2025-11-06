"""
Generate Sample LLM Embeddings for Testing
===========================================
This script creates sample embeddings simulating real LLM instruction data.
Useful for testing the PPOv1 pipeline without needing actual LLM APIs.
"""

import numpy as np
import argparse
import os


def generate_sample_instruction_embeddings(n_samples=1000, embedding_dim=384, n_clusters=5, seed=42):
    """
    Generate sample embeddings that mimic real instruction-tuning data.
    
    Creates clustered embeddings representing different instruction types:
    - Cluster 1: Math/reasoning instructions
    - Cluster 2: Creative writing instructions
    - Cluster 3: Code generation instructions
    - Cluster 4: Q&A instructions
    - Cluster 5: Translation instructions
    
    Args:
        n_samples: Number of instruction embeddings to generate
        embedding_dim: Dimension of embeddings (384 for MiniLM, 768 for MPNet, etc.)
        n_clusters: Number of instruction type clusters
        seed: Random seed for reproducibility
    
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        cluster_labels: cluster assignment for each sample
    """
    rng = np.random.default_rng(seed)
    
    # Create diverse cluster centers
    cluster_centers = rng.normal(0, 2.0, size=(n_clusters, embedding_dim))
    
    # Make clusters more distinct (simulate different instruction types)
    for i in range(n_clusters):
        # Add some structure to make clusters more realistic
        cluster_centers[i] += i * 0.5  # Offset each cluster
        cluster_centers[i] *= (1 + i * 0.2)  # Scale differently
    
    embeddings = []
    cluster_labels = []
    
    # Generate samples
    for i in range(n_samples):
        # Assign to cluster (some imbalance like real data)
        cluster_probs = np.array([0.25, 0.15, 0.30, 0.20, 0.10])  # Imbalanced like real data
        cluster_id = rng.choice(n_clusters, p=cluster_probs)
        
        # Sample from cluster with noise
        center = cluster_centers[cluster_id]
        noise = rng.normal(0, 0.3, size=embedding_dim)
        embedding = center + noise
        
        # Normalize (like real embeddings)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        embeddings.append(embedding)
        cluster_labels.append(cluster_id)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    cluster_labels = np.array(cluster_labels, dtype=np.int32)
    
    return embeddings, cluster_labels


def generate_sample_texts(n_samples=1000, seed=42):
    """
    Generate sample instruction texts (for reference/visualization).
    """
    rng = np.random.default_rng(seed)
    
    templates = {
        0: [  # Math/reasoning
            "Solve the equation: {}",
            "Calculate: {}",
            "What is the probability of {}?",
            "Prove that {}",
        ],
        1: [  # Creative writing
            "Write a poem about {}",
            "Create a story featuring {}",
            "Describe {} in vivid detail",
            "Write a dialogue between {}",
        ],
        2: [  # Code
            "Write Python code to {}",
            "Debug this code: {}",
            "Implement {} in JavaScript",
            "Optimize this function: {}",
        ],
        3: [  # Q&A
            "What is {}?",
            "Explain {} in simple terms",
            "Why does {}?",
            "How can I {}?",
        ],
        4: [  # Translation
            "Translate to French: {}",
            "Convert this to Spanish: {}",
            "Rephrase: {}",
            "Summarize: {}",
        ]
    }
    
    topics = [
        "machine learning", "climate change", "quantum physics", "cooking",
        "history", "music", "art", "mathematics", "biology", "technology"
    ]
    
    texts = []
    cluster_probs = np.array([0.25, 0.15, 0.30, 0.20, 0.10])
    
    for i in range(n_samples):
        cluster_id = rng.choice(5, p=cluster_probs)
        template = rng.choice(templates[cluster_id])
        topic = rng.choice(topics)
        text = template.format(topic)
        texts.append(text)
    
    return texts


def main():
    parser = argparse.ArgumentParser(description="Generate sample LLM embeddings")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--embedding-dim", type=int, default=384,
                        help="Embedding dimension (384 for MiniLM, 768 for MPNet)")
    parser.add_argument("--output-dir", type=str, default="sample_embeddings",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Generating Sample LLM Instruction Embeddings")
    print("="*60)
    print(f"Number of samples: {args.n_samples}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate embeddings
    print("\n📊 Generating embeddings...")
    embeddings, cluster_labels = generate_sample_instruction_embeddings(
        n_samples=args.n_samples,
        embedding_dim=args.embedding_dim,
        seed=args.seed
    )
    
    # Generate sample texts
    print("📝 Generating sample instruction texts...")
    texts = generate_sample_texts(n_samples=args.n_samples, seed=args.seed)
    
    # Save embeddings
    embeddings_path = os.path.join(args.output_dir, "sample_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"\n✓ Saved embeddings to: {embeddings_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    
    # Save cluster labels
    labels_path = os.path.join(args.output_dir, "cluster_labels.npy")
    np.save(labels_path, cluster_labels)
    print(f"✓ Saved cluster labels to: {labels_path}")
    
    # Save sample texts
    texts_path = os.path.join(args.output_dir, "sample_texts.txt")
    with open(texts_path, 'w') as f:
        for i, text in enumerate(texts[:100]):  # Save first 100
            f.write(f"{i}: {text}\n")
    print(f"✓ Saved sample texts (first 100) to: {texts_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Embedding Statistics")
    print(f"{'='*60}")
    print(f"Mean: {embeddings.mean():.4f}")
    print(f"Std: {embeddings.std():.4f}")
    print(f"Min: {embeddings.min():.4f}")
    print(f"Max: {embeddings.max():.4f}")
    
    # Print cluster distribution
    print(f"\n{'='*60}")
    print("Cluster Distribution (simulating instruction types)")
    print(f"{'='*60}")
    cluster_names = [
        "Math/Reasoning",
        "Creative Writing",
        "Code Generation",
        "Q&A",
        "Translation"
    ]
    for i in range(5):
        count = np.sum(cluster_labels == i)
        pct = count / len(cluster_labels) * 100
        print(f"Cluster {i} ({cluster_names[i]:<20}): {count:4d} samples ({pct:5.1f}%)")
    
    # Print usage example
    print(f"\n{'='*60}")
    print("Usage Example")
    print(f"{'='*60}")
    print(f"\nTrain PPOv1 with these embeddings:")
    print(f"\nuv run python main3.py \\")
    print(f"    --embeddings-path {embeddings_path} \\")
    print(f"    --target-subset-size 100 \\")
    print(f"    --total-timesteps 10000")
    
    print(f"\nOr test with comparison:")
    print(f"\nuv run python compare_ppo_versions.py")
    
    print(f"\n{'='*60}")
    print("✅ Sample embeddings generated successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


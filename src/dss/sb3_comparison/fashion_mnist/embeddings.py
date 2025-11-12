import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def flatten_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.view(-1)

def build_fashion_mnist_embeddings(batch_size: int=512, device: str | None = None):
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.2860,), (0.3530,)),
      transforms.Lambda(flatten_tensor)
  ])

  ds = datasets.FashionMNIST(
      root="./data",
      train=True,
      download=True,
      transform=transform
  )

  loader = DataLoader(
      ds,
      batch_size=batch_size,
      shuffle=False,
      num_workers=4,
      pin_memory=True
  )

  X_list, y_list = [], []
  for imgs, labels in loader:
    X_list.append(imgs.float().cpu().numpy())
    y_list.append(labels.long().cpu().numpy())

  X_embed = np.concatenate(X_list, axis=0).astype(np.float32)
  y = np.concatenate(y_list, axis=0).astype(np.int64)

  return X_embed, y
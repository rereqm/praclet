from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset.loader import GTSDBDataset, collate_fn

root = Path(__file__).resolve().parents[2]

dataset = GTSDBDataset(
    root / "data/processed/images/train",
    root / "data/processed/labels/train"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn
)

images, targets = next(iter(loader))

print(len(images))
print(images[0].shape)
print(targets[0])
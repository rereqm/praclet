from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset.loader import GTSDBDataset, collate_fn
from src.models.faster_rcnn import get_model


def main():

    root = Path(__file__).resolve().parents[2]

    train_dataset = GTSDBDataset(
        root / "data/processed/images/train",
        root / "data/processed/labels/train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    device = torch.device("cuda")

    model = get_model().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    epochs = 100

    model.train()

    for epoch in range(epochs):

        epoch_loss = 0

        for images, targets in train_loader:

            images = [img.to(device) for img in images]

            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)

            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss = {epoch_loss:.3f}")

    save_path = root / "results" / "weights" / "faster_rcnn.pth"

    torch.save(model.state_dict(), save_path)

    print("\nОбучение завершено.")
    print(f"Модель сохранена: {save_path}")


if __name__ == "__main__":
    main()
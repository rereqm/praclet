from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset.loader import GTSDBDataset, collate_fn
from src.models.ssd import get_model


def main():

    root = Path(__file__).resolve().parents[2]

    train_dataset = GTSDBDataset(
        root / "data/processed/images/train",
        root / "data/processed/labels/train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,          # RTX 4060 Ti должна спокойно выдержать
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Используется устройство: {device}")

    # После создания модели
    model = get_model().to(device)

    # Явная проверка и повторное перемещение (на всякий случай)
    print(f"Параметры модели на: {next(model.parameters()).device}")
    if next(model.parameters()).device != device:
        model = model.to(device)
        print("Модель принудительно перемещена на", device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
        momentum=0.9,
        weight_decay=5e-4
    )

    epochs = 100

    model.train()

    for epoch in range(epochs):

        epoch_loss = 0.0

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

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Loss = {epoch_loss / len(train_loader):.4f}"
        )

    save_dir = root / "results" / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "ssd300.pth"

    torch.save(model.state_dict(), save_path)

    print("\nОбучение завершено.")
    print(f"Модель сохранена: {save_path}")


if __name__ == "__main__":
    main()
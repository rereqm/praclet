from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class GTSDBDataset(Dataset):

    def __init__(self, images_dir, labels_dir, image_size=300):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        self.images = sorted(list(self.images_dir.glob("*.png")))
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # -------------------------
        # Resize (ВАЖНО для SSD)
        # -------------------------
        image = image.resize((self.image_size, self.image_size))
        image = F.to_tensor(image)

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())

                    # YOLO → absolute coords (orig image)
                    x1 = (xc - bw / 2) * orig_w
                    y1 = (yc - bh / 2) * orig_h
                    x2 = (xc + bw / 2) * orig_w
                    y2 = (yc + bh / 2) * orig_h

                    # scale to resized image
                    x1 *= self.image_size / orig_w
                    x2 *= self.image_size / orig_w
                    y1 *= self.image_size / orig_h
                    y2 *= self.image_size / orig_h

                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)  # background = 0

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # ⚠️ SSD не любит пустые targets
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
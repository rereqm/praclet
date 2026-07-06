from pathlib import Path
from collections import defaultdict
from PIL import Image
import shutil
import random
from configs.classes import CLASSES



class GTSDBConverter:
    def __init__(self):
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent

        # Теперь все пути строим от project_root
        self.raw_dir = project_root / "data" / "raw" / "GTSDB"
        self.output_dir = project_root / "data" / "processed"

        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        self.gt_file = self.raw_dir / "gt.txt"

    def create_directories(self):
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

    def load_annotations(self):

        annotations = defaultdict(list)

        with open(self.gt_file, "r") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                filename, x1, y1, x2, y2, cls = line.split(";")

                annotations[filename].append(
                    (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        int(cls)
                    )
                )

        return annotations

    def convert_bbox_to_yolo(self, x1, y1, x2, y2, image_width, image_height):

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        width = x2 - x1
        height = y2 - y1

        x_center /= image_width
        y_center /= image_height

        width /= image_width
        height /= image_height

        return (
            x_center,
            y_center,
            width,
            height
        )

    def split_dataset(self, annotations):

        image_names = list(annotations.keys())

        random.seed(42)
        random.shuffle(image_names)

        total = len(image_names)

        train_end = int(total * 0.7)
        val_end = int(total * 0.9)

        train = image_names[:train_end]
        val = image_names[train_end:val_end]
        test = image_names[val_end:]

        return train, val, test

    def save_split(self, image_list, split_name, annotations):

        images_output = self.images_dir / split_name
        labels_output = self.labels_dir / split_name

        for image_name in image_list:

            image_path = self.raw_dir / image_name

            image_output_path = images_output / Path(image_name).with_suffix(".png")
            label_output_path = labels_output / Path(image_name).with_suffix(".txt")

            with Image.open(image_path) as image:

                image = image.convert("RGB")

                image_width, image_height = image.size

                image.save(image_output_path)

                with open(label_output_path, "w") as label_file:
                    for x1, y1, x2, y2, cls in annotations[image_name]:
                        x_center, y_center, width, height = self.convert_bbox_to_yolo(
                            x1,
                            y1,
                            x2,
                            y2,
                            image_width,
                            image_height
                        )

                        label_file.write(
                            f"{cls} "
                            f"{x_center:.6f} "
                            f"{y_center:.6f} "
                            f"{width:.6f} "
                            f"{height:.6f}\n"
                        )

    def create_dataset_yaml(self):
        yaml_path = self.output_dir / "dataset.yaml"

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {self.output_dir.resolve()}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n\n")

            f.write("names:\n")

            for idx, name in enumerate(CLASSES):
                f.write(f"  {idx}: {name}\n")

    def run(self):

        self.create_directories()

        annotations = self.load_annotations()

        train, val, test = self.split_dataset(annotations)

        self.save_split(train, "train", annotations)
        self.save_split(val, "val", annotations)
        self.save_split(test, "test", annotations)

        self.create_dataset_yaml()

        print("Датасет успешно подготовлен!")

if __name__ == "__main__":
    converter = GTSDBConverter()
    converter.run()
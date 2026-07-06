import time
import torch



def evaluate_model(model, dataloader, device):
    model.eval()

    total_time = 0.0
    num_images = 0

    all_predictions = []

    with torch.no_grad():
        for images, targets in dataloader:

            images = [img.to(device) for img in images]

            start = time.perf_counter()

            outputs = model(images)

            end = time.perf_counter()

            total_time += (end - start)
            num_images += len(images)

            all_predictions.append(outputs)

    fps = num_images / total_time if total_time > 0 else 0

    return {
        "fps": fps,
        "num_images": num_images
    }
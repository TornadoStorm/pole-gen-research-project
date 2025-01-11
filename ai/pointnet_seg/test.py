import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.string import format_accuracy

from .model import PointNetSeg


# TODO More metrics
def test(model: PointNetSeg, test_data: DataLoader):
    model.eval()

    acc = 0.0
    iou = 0.0
    num_batches = len(test_data)

    with torch.no_grad():
        for data in tqdm(test_data, desc="Testing", total=num_batches):
            inputs, labels = data
            inputs = inputs.float()
            preds = model(inputs)

            acc += model.accuracy(preds, labels)
            iou += model.iou(preds, labels)

    acc /= num_batches
    iou /= num_batches

    print(f"Accuracy: {format_accuracy(acc)}")
    print(f"IoU: {iou}")

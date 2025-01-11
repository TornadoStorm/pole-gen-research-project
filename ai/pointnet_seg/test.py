import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.string import format_accuracy

from .model import PointNetSeg


def test(model: PointNetSeg, test_data: DataLoader):
    model.eval()
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for data in tqdm(test_data, desc="Testing", total=len(test_data)):
            inputs, labels = data
            inputs = inputs.float()
            outputs, __, __ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()

    accuracy: float = correct / total
    print(f"Test accuracy: {format_accuracy(accuracy)}")

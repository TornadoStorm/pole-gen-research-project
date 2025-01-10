import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.string import format_accuracy

from .model import PointNetSeg


def test(model: PointNetSeg, dataloader: DataLoader):
    model.eval()
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for data in tqdm(dataloader, dec="Testing"):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels
            outputs, __, __ = model(inputs.transpose(1, 2))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()

    accuracy: float = correct / total
    print(f"Test accuracy: {format_accuracy(accuracy)}")

import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from utils.string import format_accuracy

from .data_processing import PointCloudData
from .model import PointNetSeg


def pointNetLoss(ouputs, labels, m3x3, m64x64, alpha=0.0001) -> torch.nn.Module:
    criterion = torch.nn.NLLLoss()
    bs = ouputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if ouputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(ouputs, labels) + alpha * (
        torch.norm(diff3x3) + torch.norm(diff64x64)
    ) / float(bs)


def train(
    pointnet: PointNetSeg,
    optimizer: torch.optim.Optimizer,
    train_data: DataLoader,
    eval_data: DataLoader | None = None,
    epochs: int = 15,
    out_dir: str | None = None,
):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        # Empty the directory
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))

    # TODO store loss and accuracy for graph

    for epoch in trange(epochs, desc="Training", unit="epoch", position=0, leave=True):
        pointnet.train()
        running_loss = 0.0

        for i, data in tqdm(
            enumerate(train_data, 0),
            desc=f"Epoch ${epoch + 1}/{epochs}",
            unit="batch",
            position=epoch + 1,
            leave=True,
        ):
            inputs, labels = data
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            loss = pointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9 or True:
                print(f"Epoch {epoch + 1} batch {i + 1} loss: {running_loss / 10}")
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # Validation
        with torch.no_grad():
            for data in eval_data:
                inputs, labels = data
                inputs = inputs.float()
                labels = labels
                outputs, __, __ = pointnet(inputs.transpose(1, 2))
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        # Save the model
        if out_dir is not None:
            path = os.path.join(
                out_dir, f"pointnet_seg_{epoch:03d}_{(val_acc * 100.0):.2f}.pth"
            )
            print(f"Epoch {epoch} accuracy: {format_accuracy(val_acc)}")
            torch.save(pointnet.state_dict(), path)


if __name__ == "__main__":
    pointnet = PointNetSeg()
    # pointnet.to(device)
    dataset_path = "dataset"
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.005)
    train_ds = PointCloudData(dataset_path, start=0, end=100)
    val_ds = PointCloudData(dataset_path, start=100, end=120)
    # warning: batch_size needs to be at least 2
    train_loader = DataLoader(dataset=train_ds, batch_size=5, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=5, shuffle=False)
    train(pointnet, optimizer, train_loader, val_loader, save=True)

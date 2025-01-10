import os

import torch
from torch.utils.data import DataLoader

from .data_processing import PointCloudData
from .model import PointNetSeg


def pointNetLoss(ouputs, labels, m3x3, m64x64, alpha=0.0001):
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
    best_val_acc = -1.0
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0

        for i, data in enumerate(train_data, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            loss = pointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9 or True:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))
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

        print("correct", correct, "/", total)
        val_acc = 100.0 * correct / total
        print("Valid accuracy: %d %%" % val_acc)

        # Save the model
        if out_dir is not None and val_acc > best_val_acc:
            os.makedirs(out_dir, exist_ok=True)
            best_val_acc = val_acc
            path = os.path.join(out_dir, "pointnetmodel.pth")
            print("best_val_acc:", val_acc, "saving model at", path)
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

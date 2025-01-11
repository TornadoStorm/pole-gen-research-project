import torch
import torch.utils
from fastai.vision.all import *
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader

from .model import PointNetSeg


def accuracy(preds, labels):
    outputs, _, _ = preds
    preds = outputs.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return (preds == labels).mean()


def iou(preds, labels):
    outputs, _, _ = preds
    preds = outputs.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return jaccard_score(preds.flatten(), labels.flatten(), average="macro")


def pointNetLoss(preds, labels, alpha=0.0001) -> torch.nn.Module:
    outputs, m3x3, m64x64 = preds
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (
        torch.norm(diff3x3) + torch.norm(diff64x64)
    ) / float(bs)


def train(
    model: PointNetSeg,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    epochs: int = 15,
    batch_size: int = 32,
    num_workers: int = 4,
    model_path: str = "pointnet_seg/bestmodel",
    log_path: str = "pointnet_seg/log.csv",
):
    learn = Learner(
        DataLoaders(
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        ),
        model,
        loss_func=pointNetLoss,
        opt_func=Adam,
        metrics=[accuracy, iou],
        cbs=[
            ShowGraphCallback(),
            SaveModelCallback(fname=model_path),
            CSVLogger(fname=log_path),
        ],
    )

    learn.fit(epochs)

import torch
import torch.utils
from fastai.vision.all import *
from torch.utils.data import DataLoader

from .model import PointNetSeg


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
        loss_func=PointNetSeg.loss,
        opt_func=Adam,
        metrics=[PointNetSeg.accuracy, PointNetSeg.iou],
        cbs=[
            ShowGraphCallback(),
            SaveModelCallback(fname=model_path),
            CSVLogger(fname=log_path),
        ],
    )

    learn.fit(epochs)

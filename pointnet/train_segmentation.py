from __future__ import print_function

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from pointnet.model import PointNetDenseCls, feature_transform_regularizer


def train_segmentation(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    class_choice: int,
    k: int,
    batchSize: int = 32,
    workers: int = 4,
    epochs: int = 25,
    outf: str = "seg",
    model: str = "",
    feature_transform: bool = False,
):
    manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchSize, shuffle=True, num_workers=workers
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchSize, shuffle=True, num_workers=workers
    )

    try:
        os.makedirs(outf)
    except OSError:
        pass

    blue = lambda x: "\033[94m" + x + "\033[0m"

    classifier = PointNetDenseCls(k=k, feature_transform=feature_transform)

    if model != "":
        classifier.load_state_dict(torch.load(model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    num_batch = len(train_dataset) / batchSize

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, k)
            target = target.view(-1, 1)[:, 0] - 1
            # print(pred.size(), target.size())
            loss = F.nll_loss(pred, target)
            if feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print(
                "[%d: %d/%d] train loss: %f accuracy: %f"
                % (
                    epoch,
                    i,
                    num_batch,
                    loss.item(),
                    correct.item() / float(batchSize * 2500),
                )
            )

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                pred = pred.view(-1, k)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print(
                    "[%d: %d/%d] %s loss: %f accuracy: %f"
                    % (
                        epoch,
                        i,
                        num_batch,
                        blue("test"),
                        loss.item(),
                        correct.item() / float(batchSize * 2500),
                    )
                )

        scheduler.step()

        torch.save(
            classifier.state_dict(),
            "%s/seg_model_%s_%d.pth" % (outf, class_choice, epoch),
        )

    ## benchmark mIOU
    shape_ious = []
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(k)  # np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(
                    np.logical_and(
                        pred_np[shape_idx] == part, target_np[shape_idx] == part
                    )
                )
                U = np.sum(
                    np.logical_or(
                        pred_np[shape_idx] == part, target_np[shape_idx] == part
                    )
                )
                if U == 0:
                    iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print(f"mIOU for class {class_choice}: {np.mean(shape_ious)}")

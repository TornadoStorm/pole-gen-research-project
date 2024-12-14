from __future__ import print_function

import os
import random
import warnings
from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from utils.string import format_accuracy

from .model import PointNetCls, feature_transform_regularizer


def train_classifier(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    k: int = 40,
    batchSize: int = 32,
    workers: int = 4,
    epochs: int = 25,
    outf: str = "cls",
    model: str = "",
    feature_transform: bool = False,
) -> PointNetCls:

    manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchSize, shuffle=True, num_workers=workers
    )

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=workers,
    )

    try:
        os.makedirs(outf)
    except OSError:
        pass

    classifier = PointNetCls(k=k, feature_transform=feature_transform)

    # Decide on whether to use GPU or CPU
    can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if can_cuda else "cpu")
    if not can_cuda:
        warnings.warn("CUDA is not available. Training will be slower.")
    classifier.to(device)

    if model != "":
        classifier.load_state_dict(torch.load(model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    num_batch = len(train_dataset) / batchSize

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print(
                f"[{epoch}: {i}/{num_batch}] train loss: {loss.item()} accuracy: {format_accuracy(correct.item() / float(batchSize))}"
            )

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print(
                    f"[{epoch}: {i}/{num_batch}] \033[94mtest\033[0m loss: {loss.item()} accuracy: {format_accuracy(correct.item() / float(batchSize))}"
                )

        scheduler.step()

        torch.save(classifier.state_dict(), "%s/cls_model_%d.pth" % (outf, epoch))

    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print(f"final accuracy {format_accuracy(total_correct / float(total_testset))}")

    return classifier


def evaluate_classifier(
    classifier: PointNetCls,
    test_dataset: torch.utils.data.Dataset,
):
    classifier.eval()
    predictions: List[bool] = []
    predictions_by_class: defaultdict[int, List[bool]] = defaultdict(list)
    with torch.no_grad():
        with tqdm(desc="Evaluating classifier", total=len(test_dataset)) as pbar:
            for entry in test_dataset:
                input_data = entry[0].unsqueeze(0)
                input_data = input_data.transpose(1, 2)
                scores = classifier(input_data)[0][0]
                pre = scores.argmax().item()
                exp = int(entry[1])
                predictions.append(pre == exp)
                predictions_by_class[exp].append(pre == exp)
                pbar.update(1)

    accuracy = sum(predictions) / len(predictions)
    accuracy_by_class = {k: sum(v) / len(v) for k, v in predictions_by_class.items()}

    return accuracy, accuracy_by_class

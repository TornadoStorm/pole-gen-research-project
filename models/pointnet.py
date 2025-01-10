from typing import Literal

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


class TNETkd(nn.Module):
    def __init__(self, k=64):
        super(TNETkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = TNETkd(k=3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = TNETkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet(nn.Module):
    def __init__(self, task, k=2, feature_transform=False):
        super(PointNet, self).__init__()

        if task == "cls":
            self.model = PointNetCls(k, feature_transform)
        elif task == "seg":
            self.model = PointNetDenseCls(k, feature_transform)

    def forward(self, x):
        return self.model(x)


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batch_size = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


class PointNetLightningModel(L.LightningModule):
    def __init__(
        self,
        task: Literal["cls", "seg"],
        num_classes: int,
        learning_rate: float = 0.001,
        pretrain_path="",
        feature_transform=False,
    ):
        super(PointNetLightningModel, self).__init__()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.task = task
        self.lr = learning_rate
        self.num_classes = num_classes
        self.feature_transform = feature_transform

        self.pointnet = PointNet(
            self.task, self.num_classes, feature_transform=self.feature_transform
        )
        if pretrain_path != "":
            self.pointnet.load_state_dict(torch.load(pretrain_path))

    def forward(self, points):
        points = points.transpose(2, 1)
        pred, trans, trans_feat = self.pointnet(points)
        return pred, trans, trans_feat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = {
            "scheduler": StepLR(optimizer, step_size=20, gamma=0.5),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        points, target = batch
        pred, trans, trans_feat = self(points)

        if self.task == "cls":
            target = target[:, 0]
        elif self.task == "seg":
            pred = pred.view(-1, self.num_classes)
            target = target.view(-1, 1)[:, 0] - 1

        loss = F.nll_loss(pred, target)
        if self.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        pred_choice = pred.data.max(1)[1]
        acc = pred_choice.eq(target.data).cpu().sum()

        result = {"loss": loss, "acc": acc}
        self.training_step_outputs.append(result)
        return result  # , pred

    def on_train_epoch_end(self):
        loss = (
            torch.hstack([output["loss"] for output in self.training_step_outputs])
            .float()
            .mean()
        )
        acc = (
            torch.hstack([output["acc"] for output in self.training_step_outputs])
            .float()
            .mean()
        )
        self.log("train loss", loss)
        self.log("train acc", acc)

    def validation_step(self, batch, batch_idx):
        points, target = batch
        # target = target[:, 0]
        pred, trans, trans_feat = self(points)

        if self.task == "cls":
            target = target[:, 0]
        elif self.task == "seg":
            pred = pred.view(-1, self.num_classes)
            target = target.view(-1, 1)[:, 0] - 1

        loss = F.nll_loss(pred, target)
        if self.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        pred_choice = pred.data.max(1)[1]
        acc = pred_choice.eq(target.data).cpu().sum()

        result = {"loss": loss, "acc": acc}
        self.validation_step_outputs.append(result)
        return result  # , pred

    def on_validation_epoch_end(self) -> None:
        loss = (
            torch.hstack([output["loss"] for output in self.validation_step_outputs])
            .float()
            .mean()
        )
        acc = (
            torch.hstack([output["acc"] for output in self.validation_step_outputs])
            .float()
            .mean()
        )
        self.log("val loss", loss)
        self.log("val acc", acc)

    def test_step(self, batch, batch_idx):
        points, target = batch
        # target = target[:, 0]
        pred, trans, trans_feat = self(points)

        if self.task == "cls":
            target = target[:, 0]
        elif self.task == "seg":
            pred = pred.view(-1, self.num_classes)
            target = target.view(-1, 1)[:, 0] - 1

        loss = F.nll_loss(pred, target)
        if self.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        pred_choice = pred.data.max(1)[1]
        acc = pred_choice.eq(target.data).cpu().sum()

        result = {"loss": loss, "acc": acc}
        self.test_step_outputs.append(result)
        return result  # , pred

    def on_test_epoch_end(self) -> None:
        loss = (
            torch.hstack([output["loss"] for output in self.test_step_outputs])
            .float()
            .mean()
        )
        acc = (
            torch.hstack([output["acc"] for output in self.test_step_outputs])
            .float()
            .mean()
        )
        self.log("test loss", loss)
        self.log("test acc", acc)

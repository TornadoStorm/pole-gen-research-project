# pyTorch imports
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score


# Multi Layer Perceptron
class MLP_CONV(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv = nn.Conv1d(self.input_size, self.output_size, 1)
        self.bn = nn.BatchNorm1d(self.output_size)

    def forward(self, input):
        return F.relu(self.bn(self.conv(input)))


# Fully Connected with Batch Normalization
class FC_BN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lin = nn.Linear(self.input_size, self.output_size)
        self.bn = nn.BatchNorm1d(self.output_size)

    def forward(self, input):
        return F.relu(self.bn(self.lin(input)))


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.mlp1 = MLP_CONV(self.k, 64)
        self.mlp2 = MLP_CONV(64, 128)
        self.mlp3 = MLP_CONV(128, 1024)

        self.fc_bn1 = FC_BN(1024, 512)
        self.fc_bn2 = FC_BN(512, 256)

        self.fc3 = nn.Linear(256, k * k)

    def forward(self, input):
        # input.shape == (batch_size,n,3)

        bs = input.size(0)
        xb = self.mlp1(input)
        xb = self.mlp2(xb)
        xb = self.mlp3(xb)

        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)

        xb = self.fc_bn1(flat)
        xb = self.fc_bn2(xb)

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        self.mlp1 = MLP_CONV(3, 64)
        self.mlp2 = MLP_CONV(64, 128)

        # 1D convolutional layer with kernel size 1
        self.conv = nn.Conv1d(128, 1024, 1)

        # Batch normalization for stability and faster training
        self.bn = nn.BatchNorm1d(1024)

    def forward(self, input):
        n_pts = input.size()[2]
        matrix3x3 = self.input_transform(input)
        input_transform_output = torch.bmm(
            torch.transpose(input, 1, 2), matrix3x3
        ).transpose(1, 2)
        x = self.mlp1(input_transform_output)
        matrix64x64 = self.feature_transform(x)
        feature_transform_output = torch.bmm(
            torch.transpose(x, 1, 2), matrix64x64
        ).transpose(1, 2)
        x = self.mlp2(feature_transform_output)
        x = self.bn(self.conv(x))
        global_feature = nn.MaxPool1d(x.size(-1))(x)
        global_feature_repeated = (
            nn.Flatten(1)(global_feature)
            .repeat(n_pts, 1, 1)
            .transpose(0, 2)
            .transpose(0, 1)
        )

        return (
            [feature_transform_output, global_feature_repeated],
            matrix3x3,
            matrix64x64,
        )


class PointNetSeg(L.LightningModule):
    def __init__(self, n_classes: int):
        super().__init__()
        self.pointnet = PointNet()
        self.mlp1 = MLP_CONV(1088, 512)
        self.mlp2 = MLP_CONV(512, 256)
        self.mlp3 = MLP_CONV(256, 128)
        self.conv = nn.Conv1d(128, n_classes, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.criterion = torch.nn.NLLLoss()
        self.save_hyperparameters()

    def forward(self, input):
        inputs, matrix3x3, matrix64x64 = self.pointnet(input.transpose(1, 2))
        stack = torch.cat(inputs, 1)
        x = self.mlp1(stack)
        x = self.mlp2(x)
        x = self.mlp3(x)
        output = self.conv(x)
        return self.logsoftmax(output), matrix3x3, matrix64x64

    def training_step(self, batch, batch_idx):
        input, labels = batch
        preds = self(input)
        loss = self.loss(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log_stats(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.log_stats(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    # ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤
    # ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ Utility functions ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤
    # ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤ ◢◤

    def log_stats(self, batch, batch_idx, prefix):
        input, labels = batch
        preds = self(input)
        loss = self.loss(preds, labels)
        acc = self.accuracy(preds, labels)
        iou = self.iou(preds, labels)
        self.log_dict(
            {
                f"{prefix}_loss": loss,
                f"{prefix}_acc": acc,
                f"{prefix}_iou": iou,
            }
        )

    def loss(self, preds, labels, alpha=0.0001):
        outputs, m3x3, m64x64 = preds
        bs = outputs.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
        if outputs.is_cuda:
            id3x3 = id3x3.cuda()
            id64x64 = id64x64.cuda()
        diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
        diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
        return self.criterion(outputs, labels) + alpha * (
            torch.norm(diff3x3) + torch.norm(diff64x64)
        ) / float(bs)

    def accuracy(self, preds, labels):
        outputs, _, _ = preds
        preds = outputs.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        return (preds == labels).mean()

    def iou(self, preds, labels):
        outputs, _, _ = preds
        preds = outputs.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        return jaccard_score(preds.flatten(), labels.flatten(), average="macro")

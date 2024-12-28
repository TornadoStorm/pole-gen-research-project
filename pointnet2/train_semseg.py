import time
import warnings

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from .provider import rotate_point_cloud_z


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def train(
    train_dataset: data.Dataset,
    test_dataset: data.Dataset,
    num_classes: int,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    batch_size: int = 16,
    epochs: int = 32,
    learning_rate: float = 0.001,
    decay_rate: float = 1e-4,
    num_points: int = 4096,
    step_size: int = 10,
    lr_decay: float = 0.7,
    test_area: int = 5,
    out_dir: str = "seg",
):
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
    )
    weights = torch.Tensor(train_dataset.labelweights).cuda()

    classifier = model.cuda()

    can_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if can_cuda else "cpu")
    if not can_cuda:
        warnings.warn("CUDA is not available. Training will be slower.")
    classifier.to(device)

    criterion = loss.cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    classifier = classifier.apply(weights_init)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=decay_rate,
    )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, epochs):
        """Train on chopped scenes"""
        print("**** Epoch %d (%d/%s) ****" % (global_epoch + 1, epoch + 1, epochs))
        lr = max(
            learning_rate * (lr_decay ** (epoch // step_size)),
            LEARNING_RATE_CLIP,
        )
        print("Learning rate:%f" % lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // step_size))
        if momentum < 0.01:
            momentum = 0.01
        print("BN momentum updated to: %f" % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(
            enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9
        ):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, num_classes)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += batch_size * num_points
            loss_sum += loss
        print("Training mean loss: %f" % (loss_sum / num_batches))
        print("Training accuracy: %f" % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            print("Save model...")
            savepath = str(out_dir) + "/model.pth"
            print("Saving at %s" % savepath)
            state = {
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, savepath)
            print("Saving model....")

        """Evaluate on chopped scenes"""
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(num_classes)
            total_seen_class = [0 for _ in range(num_classes)]
            total_correct_class = [0 for _ in range(num_classes)]
            total_iou_deno_class = [0 for _ in range(num_classes)]
            classifier = classifier.eval()

            print("---- EPOCH %03d EVALUATION ----" % (global_epoch + 1))
            for i, (points, target) in tqdm(
                enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9
            ):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, num_classes)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += batch_size * num_points
                tmp, _ = np.histogram(batch_label, range(num_classes + 1))
                labelweights += tmp

                for l in range(num_classes):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum(
                        (pred_val == l) & (batch_label == l)
                    )
                    total_iou_deno_class[l] += np.sum(
                        ((pred_val == l) | (batch_label == l))
                    )

            labelweights = labelweights.astype(np.float32) / np.sum(
                labelweights.astype(np.float32)
            )
            mIoU = np.mean(
                np.array(total_correct_class)
                / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
            )
            print("eval mean loss: %f" % (loss_sum / float(num_batches)))
            print("eval point avg class IoU: %f" % (mIoU))
            print("eval point accuracy: %f" % (total_correct / float(total_seen)))
            print(
                "eval point avg class acc: %f"
                % (
                    np.mean(
                        np.array(total_correct_class)
                        / (np.array(total_seen_class, dtype=np.float) + 1e-6)
                    )
                )
            )

            iou_per_class_str = "------- IoU --------\n"
            for l in range(num_classes):
                iou_per_class_str += f"class {l} weight: {labelweights[l]:.3f}, IoU: {total_correct_class[l] / float(total_iou_deno_class[l]):.3f}\n"

            print(iou_per_class_str)
            print("Eval mean loss: %f" % (loss_sum / num_batches))
            print("Eval accuracy: %f" % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                print("Save model...")
                savepath = str(out_dir) + "/best_model.pth"
                print("Saving at %s" % savepath)
                state = {
                    "epoch": epoch,
                    "class_avg_iou": mIoU,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
                print("Saving model....")
            print("Best mIoU: %f" % best_iou)
        global_epoch += 1

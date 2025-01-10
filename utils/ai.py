import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train(
    model: LightningModule,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    epochs: int = 200,
    checkpoint_path: str = "data/checkpoints",
    workers_per_dataloader: int = 4,
):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=workers_per_dataloader,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=workers_per_dataloader,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=workers_per_dataloader,
    )

    trainer = Trainer(
        min_epochs=1,
        max_epochs=epochs,
        callbacks=[
            EarlyStopping(monitor="val_acc", patience=5),
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename="{epoch}-{val_loss:.2f}-{val_acc:.3f}",
                verbose=True,
                save_last=True,
                monitor="val loss",
                mode="min",
            ),
        ],
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader
    )

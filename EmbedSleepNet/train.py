import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import random
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint

from model import TinySleepNet, EmbedSleepNet
from lightning_wrapper import LightningWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the model')
    # parser.add_argument('--flavor', choices=['embed', 'tiny'], required=True)
    parser.add_argument('--flavor', choices='embed')
    parser.add_argument('--epochs', type=int, default=450)
    parser.add_argument('--model_name', type=str, default='model')
    args = parser.parse_args()

    random.seed(42)

    if args.flavor == 'tiny':
        net = TinySleepNet()
    else:
        net = EmbedSleepNet()

    model = LightningWrapper(net)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                          dirpath='.',
                                          filename=args.model_name,
                                          save_top_k=1,
                                          mode='max')
    gpus = 1 if torch.cuda.is_available() else 0
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback],reload_dataloaders_every_epoch=True,
                         gpus=gpus, max_epochs=args.epochs)
    trainer.fit(model)
    print(f'Saved model accuracy {model.max_acc * 100}%')

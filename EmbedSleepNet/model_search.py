import pytorch_lightning as pl
import random

from model import TinySleepNet
from lightning_wrapper import LightningWrapper
from benchmark import benchmark

random.seed(42)
results = []
for conv1_ch in [128, 96, 64, 32, 16, 8]:
    model = LightningWrapper(TinySleepNet(conv1_ch=conv1_ch))
    trainer = pl.Trainer(reload_dataloaders_every_epoch=True, gpus=1, max_epochs=150)
    trainer.fit(model)
    bench = benchmark(model.net, num_tests=1000)
    print(f"{conv1_ch} {bench} {model.max_acc} {model.best_k} {model.best_f1}")
    results.append((conv1_ch, model.max_acc, model.best_k, model.best_f1))

print('Final results:')
for entry in results:
    print(f'{entry[0]}\t{entry[1]}\t{entry[2]}\t{entry[3]}\t{entry[4]}')

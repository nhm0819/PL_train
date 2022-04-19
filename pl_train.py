# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

# Pytorch modules
import torch

# Pytorch-Lightning
import pytorch_lightning as pl

from pl_model import pl_classifier
from pl_logger import Logger
from pl_datamodule import WireDataModule
import argparse
import random
import os
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Classification')
parser.add_argument('--dataset_dir', type=str, default="/volume/data/kesco", metavar='S',
                    help='data directory path')
parser.add_argument('--model_name', type=str, default="efficientnet_b4", metavar='S',
                    help='model name in timm package (default: efficientnet_b4)')
parser.add_argument('--num_classes', type=int, default=3, metavar='N',
                    help='number of classes')
parser.add_argument('--epochs', type=int, default=800, metavar='N',
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size / num_gpus')
parser.add_argument('--checkpoint', type=int, default=5, metavar='N',
                    help='checkpoint period')
parser.add_argument('--num_workers', type=int, default=10, metavar='N',
                    help='number of workers')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')

parser.add_argument('--wandb_project', type=str, default="kesco-clf")
parser.add_argument('--wandb_name', type=str, default="1")



def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parser.parse_args()

    args.train_image_sizes = [256, 288, 320, 380]
    num_sizes = len(args.train_image_sizes)

    wandb.init()
    # config = wandb.config

    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_name, log_model="all")


    # callback lists
    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model/model-{epoch:02d}-{val_loss:.2f}'

    ealry_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=MODEL_CKPT_PATH, filename=MODEL_CKPT, save_top_k=5, mode='min')

    # setup data
    wire_data = WireDataModule(dataset_dir=args.dataset_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # setup model
    model = pl_classifier(args, lr=args.lr, pretrained=True)


    for size_idx, train_image_size in enumerate(args.train_image_sizes):
        wire_data.prepare_data()
        wire_data.setup(size_idx=size_idx)

        # grab samples to log predictions on
        samples = next(iter(wire_data.test_dataloader()))

        trainer = pl.Trainer(
            logger=wandb_logger,    # W&B integration
            log_every_n_steps=50,   # set the logging frequency
            gpus=-1,                # use all GPUs
            max_epochs=args.epochs/num_sizes*(size_idx+1), # number of epochs
            deterministic=True,     # keep it deterministic
            callbacks=[Logger(samples),
                       ealry_stop_callback,
                       checkpoint_callback], # see Callbacks section
            precision=16,
            check_val_every_n_epoch= args.checkpoint,
            strategy='ddp',
            auto_scale_batch_size=True
        )


        # fit the model
        trainer.fit(model, wire_data)

        # evaluate the model on a test set
        trainer.test(datamodule=wire_data)

    wandb.finish()



if __name__ == "__main__":
    main()

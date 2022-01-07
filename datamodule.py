from pytorch_lightning import LightningDataModule
import pandas as pd
import os
from dataset import CustomDataset
from transform import get_transforms
from torch.utils.data import DataLoader


class WireDataModule(LightningDataModule):

    def __init__(self, dataset_dir='./', batch_size=16, num_workers=0, sizes=[256,288,320,380]):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sizes = sizes

        self.train_transform = get_transforms(320, "train")
        self.test_transform = get_transforms(380, "test")
        self.train_df = pd.read_csv(os.path.join(dataset_dir, "train_data.csv"))
        self.test_df = pd.read_csv(os.path.join(dataset_dir, "test_data.csv"))

    def prepare_data(self):
        pass

    def setup(self, stage=None, size_idx=0):

        self.train_transform = get_transforms(self.sizes[size_idx], "train")

        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset(self.dataset_dir, self.train_df, transforms=self.train_transform)
            self.val_dataset = CustomDataset(self.dataset_dir, self.test_df, transforms=self.test_transform)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(self.dataset_dir, self.test_df, transforms=self.test_transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.test_dataset, batch_size=2 * self.batch_size, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=2 * self.batch_size, num_workers=self.num_workers)
        return test_loader




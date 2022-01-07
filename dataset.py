import cv2
import os
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, df, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.dataset_dir = dataset_dir


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.dataset_dir, self.df["path"][idx])
        # img_path = img_path.replace("/", "\\")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']

        label = self.df["label"][idx]

        return img, label
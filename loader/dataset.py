import cv2
import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
import albumentations



class ScenesDataset(Dataset):
  def __init__(self, df, mode, transforms=None):
    self.df = df.reset_index()
    self.mode = mode
    self.transforms = transforms
  
  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self,idx):
    row = self.df.iloc[idx]
    image = cv2.imread(row.image_path)[:,:,::-1]
    if self.transforms is not None:
        res = self.transforms(image=image)
        image = res['image'].astype(np.float32)
    else:
        image = image.astype(np.float32)

    image = image.transpose(2, 0, 1)
    if self.mode in ['train', 'val']:
        return torch.tensor(image), torch.tensor(row.class_id)
    else:
        return torch.tensor(image)

def get_transform(image_size):
  transform_train = albumentations.Compose([
      albumentations.Resize(image_size, image_size),
      albumentations.Normalize()
  ])

  transform_val = albumentations.Compose([
      albumentations.Resize(image_size, image_size),
      albumentations.Normalize()
  ])
  return transform_train, transform_val


def get_df(csvPath):
    split_data_dir = csvPath.split("/")
    data_dir = '/'.join(csvPath.split("/")[:-1])
    df = pd.read_csv(csvPath)

    if 'image_path' not in df.columns:
        df['image_path'] = df.apply(lambda x: f'{data_dir}/{split_data_dir[-1][:-4]}/{x.class_name}s/{x.image_name}', axis = 1)

    class_id2idx = {class_id: idx for idx, class_id in enumerate(sorted(df['class_name'].unique()))}
    df['class_id'] = df['class_name'].map(class_id2idx)

    out_dim = df.class_id.nunique()

    return df, out_dim
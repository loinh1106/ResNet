import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import argparse


def parse_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--train_path', type=str, required=True)
  parser.add_argument('--val_path', type=str, required=True)
  parser.add_argument('--out_path', type=str, required=True)

  args, _ = parser.parse_known_args()
  return args

def prepare_df(lst_data, out_path, id_encode):
  csv_content = []
  for img_path in lst_data:
    img_path_split = img_path.split('/')
    img_name = img_path_split[-1]
    class_name = img_path_split[-2]
    csv_content.append({
        'image_path': img_path,
        'image_name': img_name,
        'class_name': class_name,
        'class_id': id_encode[class_name]
    })
  df = pd.DataFrame(csv_content)
  df.to_csv(out_path, index=False)

if __name__ == '__main__':
    args = parse_args()
    lst_class = glob.glob(f'{args.train_path}/*')
    lst_class_val = glob.glob(f'{args.val_path}/*')
    lst_class.sort()
    lst_class_val.sort()
    lst_train, lst_val = [], []
    id_encode = {x.split('/')[-1]:idx for idx,x in enumerate(lst_class)}

    for class_name in tqdm(lst_class):
        lst_img = os.listdir(class_name)
        lst_train.extend(list(map(lambda x: f'{class_name}/{x}', lst_img)))
    
    for class_name in tqdm(lst_class_val):
        lst_img = os.listdir(class_name)
        lst_val.extend(list(map(lambda x: f"{class_name}/{x}", lst_img)))
    os.makedirs(args.out_path,exist_ok=True)
    prepare_df(lst_train, f'{args.out_path}/train.csv', id_encode)
    prepare_df(lst_val, f'{args.out_path}/val.csv', id_encode)
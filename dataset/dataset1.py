import torch
import numpy as np
import pandas as pd
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform

        self.csv_path = './dataset.csv'
        self.images = pd.read_csv(self.csv_path).values.tolist()
        # print(len(self.images))
        # print(self.images[0])

    def __getitem__(self, idx):
        img_x = Image.open(self.images[idx][0])
        img_x = img_x.resize((1692, 855))
        img_x = np.array(img_x).transpose(2, 0, 1)

        img_label = Image.open(self.images[idx][1])
        # img_label = img_label.resize((1692, 855))
        img_label = img_label.resize((1508, 660))
        img_label = np.array(img_label)

        # print(img_x.shape, img_label.shape)

        return self.transform(img_x, img_label)

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath('.'))
    from transform import create_transform

    transform = create_transform('transform1')()
    dataset = Dataset(transform)

    data_loader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = 1,#每次拼接的wav长度不一致，因此batch_size为1
            shuffle = False,
            num_workers = 0,
            collate_fn = None)

    for idx, (img_x, img_label) in enumerate(data_loader):
        print(img_x.shape, img_label.shape)
        break
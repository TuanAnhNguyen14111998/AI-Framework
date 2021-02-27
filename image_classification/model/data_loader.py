import torch
from torch.utils import data
from PIL import ImageFile
from PIL import Image
import numpy as np
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df_info, labels, image_size):
        'Initialization'
        self.labels = labels
        self.df_info = df_info
        self.image_size = image_size

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df_info)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        class_name = self.df_info.iloc[index]["class_name"]
        image_path = self.df_info.iloc[index]["image_path"]

        # Load data and get label
        image = Image.open(image_path).convert('RGB')
        image = np.array(image.resize(self.image_size)) / 255.
        X = torch.Tensor(image).permute(2, 0, 1)
        y = self.labels[class_name]

        return X, y

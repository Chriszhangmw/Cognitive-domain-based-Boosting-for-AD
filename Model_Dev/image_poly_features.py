import torch
import torch.nn as nn
import pandas as pd


class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.num_classes = num_classes
        self.ln1 = nn.Linear(197*3, 768)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(589824, self.num_classes)

    def forward(self, x):
        x = self.ln1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Equivalent to nn.Flatten()
        # print('x.shape: ',x.shape)
        x = self.ln2(x)
        return x

# Assuming you have a dataset and dataloader set up
# Replace this with your actual dataset and dataloader

import torch
from torch.utils.data import Dataset, DataLoader





# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features,labels):
        self.data = features
        self.label =labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        # Assuming each sample_data has keys 'image1', 'image2', 'image3' with list data
        image1 = torch.tensor(sample_data['image1'])
        image2 = torch.tensor(sample_data['image2'])
        image3 = torch.tensor(sample_data['image3'])
        # print('image3.shape: ',image3.shape)

        # Combine images into one tensor (assuming they have the same shape)
        # combined_image = torch.stack([image1, image2, image3], dim=0)
        combined_image = torch.cat([image1, image2, image3], dim=0)
        # print('combined_image.shape: ',combined_image.shape)
        combined_image = combined_image.view(768, -1)
        # print('combined_image.shape: ',combined_image.shape)

        # Assuming there are also corresponding labels in sample_data, replace 'label_key' with your actual key
        label = torch.tensor(self.label[idx])

        return combined_image, label



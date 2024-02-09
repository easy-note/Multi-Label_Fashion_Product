from torch.utils.data import Dataset
from utils import clean_data

import torch
import joblib
import math
import cv2
import torchvision.transforms as transforms

def train_val_split(df):
    df = clean_data(df)

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # 90% for training and 10% for validation
    num_train_samples = math.floor(len(df) * 0.90)
    num_val_samples = math.floor(len(df) * 0.10)

    train_df = df[:num_train_samples].reset_index(drop=True)
    val_df = df[-num_val_samples:].reset_index(drop=True)

    return train_df, val_df

class FashionDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.num_list_gender = joblib.load('./num_list_gender.pkl')
        self.num_list_articletype = joblib.load('./num_list_articletype.pkl')
        self.num_list_season = joblib.load('./num_list_season.pkl')
        self.num_list_usage = joblib.load('./num_list_usage.pkl')
        self.is_train = is_train

        # the training transforms and augmentations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
         # the validation transforms
        if not self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = cv2.imread(f"./data/images-all/{self.df['id'][index]}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        cat_gender = self.df['gender'][index]
        label_gender = self.num_list_gender[cat_gender]

        cat_articletype = self.df['articleType'][index]
        label_articletype = self.num_list_articletype[cat_articletype]

        cat_season = self.df['season'][index]
        label_season = self.num_list_season[cat_season]
        
        cat_usage = self.df['usage'][index]
        label_usage = self.num_list_usage[cat_usage]

        # image to float32 tensor
        image = torch.tensor(image, dtype=torch.float32)
        # labels to long tensors
        label_gender = torch.tensor(label_gender, dtype=torch.long)
        label_articletype = torch.tensor(label_articletype, dtype=torch.long)
        label_season = torch.tensor(label_season, dtype=torch.long)
        label_usage = torch.tensor(label_usage, dtype=torch.long)
        
        return {
            'image': image,
            'gender': label_gender,
            'articleType': label_articletype,
            'season': label_season,
            'usage': label_usage
        }
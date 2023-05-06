import glob
import itertools
import heapq
import json
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import pandas as pd
import yaml

from collections import OrderedDict as OrderedDict
from datetime import datetime
from tabular_data import Data_Preparation
from torchmetrics import R2Score
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        clean_data = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
        categories = clean_data['Category'].astype('category')
        # print(categories)
        encoder = pd.get_dummies(categories)
        encoder = encoder.astype('int64')
        clean_data = pd.concat([clean_data, encoder], axis=1)
        # print(clean_data.info())
        self.X, self.y = Data_Preparation.load_airbnb(clean_data, 'bedrooms')
        self.X = self.X.select_dtypes(include =['float64', 'int64'])
        print(self.X.info())
        print(self.y)

      
     
    
     



if __name__ == '__main__':
    dataset = AirbnbNightlyPriceRegressionDataset()
    


#  Run entire pipeline to train entire pipeline and find the best model.